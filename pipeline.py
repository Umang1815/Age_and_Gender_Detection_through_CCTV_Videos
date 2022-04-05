import torch
import cv2
import csv
import os
import time
import numpy as np
import argparse
import warnings
from torchvision import transforms
from statistics import mode
from PIL import Image

from ByteTracker.tracker.byte_tracker import BYTETracker

from model_arch.Swin_coral_ir import Swin_Coral
from model_arch.Efficient_coral import Eff_Coral
from model_arch.Peta_gender import MultilabelSwinL
from coral_pytorch.dataset import proba_to_label
from model_arch.swinIR import define_model, test

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


img_type = ['jpg', 'jpeg', 'png', 'gif']
vid_type = ['mp4', 'mov', 'webm', 'mkv', 'avi']

header = ['frame num',
    'person id',
    'bb_xmin',
    'bb_ymin',
    'bb_height',
    'bb_width',
    'age_min',
    'age_max',
    'age_actual',
    'gender'
]


warnings.filterwarnings('ignore')

# select device 
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# transforms required
transform_face = transforms.Compose(
            [   transforms.Resize((224, 224), transforms.InterpolationMode.BICUBIC),
                # transforms.ToTensor(),
            ]
        )
transform_body = transforms.Compose(
            [   transforms.Resize((224, 224), transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ]
        )

window_size = 8
scale = 4

def load_models(super_res=True):

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # load all models
    # yolov5

    CKPT_PATH = 'weights/crowdhuman_yolov5l.pt'
    yolov5 = torch.hub.load('yolov5','custom',
                            path=CKPT_PATH,
                            source='local',
                            force_reload=True,
                            verbose=False,
                            device=device)
    print("Yolov5 Loaded Successfully\n")

    # Multi-Headed SwinL with Coral Layer (face-age-gender)
    if super_res:
        coral_model = Swin_Coral()
        coral_model.eval().to(device)
        coral_model.load_state_dict(torch.load("weights/swin_coral_ir.pth",map_location=device))
        print("Swin Coral Loaded Successfully\n")
        
    else:
        coral_model = Eff_Coral()
        coral_model.eval().to(device)
        coral_model.load_state_dict(torch.load("weights/eff_coral_utk.pth",map_location=device))
        print("EfficientNet Coral Loaded Successfully\n")

    # Swinn L (body-gender)
    peta_body = MultilabelSwinL()
    peta_body.eval().to(device)
    peta_body.load_state_dict(torch.load("weights/peta_gender.pth",map_location=device))
    print("Swinn Gender Classifier Loaded Successfully\n")
    
    # SwinIR (upscaling)
    swinIR = define_model()
    swinIR.eval().to(device)
    print("SwinIR Loaded Successfully\n")

    return yolov5, coral_model, peta_body, swinIR



# util functions

def frame_extract(path):
    """
    faster frame extraction using yields
    """
    vidObj = cv2.VideoCapture(path) 
    success = 1

    while success:
        success, image = vidObj.read()
        if success:
            yield image


def proba_to_range(probas):
    """
    Converts predicted probabilities from extended binary format
    to range
    """
    p_low = 0.47
    p_high = 0.53
    lowlevels = probas > p_low
    highlevels = probas > p_high
    
    low_labels = torch.sum(lowlevels, dim=1)
    high_labels = torch.sum(highlevels, dim=1)
    
    return high_labels, low_labels


def bb_intersection_over_union(boxA, boxB):
    """
    Returns % area of face and body
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA ) * max(0, yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxBArea)
    return iou


def vid_pipeline(vid_path, display, save, super_res):
    # With ByteTracker without insightface only head

    tracker = BYTETracker()
    csv_data = []
    min_box_area = 50
    cap = cv2.VideoCapture(vid_path)
    frame_id = 0

    yolov5, coral_model, _, swinIR = load_models(super_res)

    vid_name = vid_path.split('/')[-1].split('.')[-2]

    if save:
        output_folder = 'outputs/'
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        size = (width,height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vid_writer = cv2.VideoWriter(output_folder+vid_name+".mp4", fourcc, 30.0, size)
        

    
    csv_folder = "csv_outputs/"
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        faces = []
        bboxes = []
        data = {}
        st = time.time()
        preds = yolov5(frame, size=640, augment=False)
        dets = preds.xyxy[0]
        # dets[:,5] = (dets[:,5] - torch.ones(dets.shape[0]).to(device))*(-1)
        online_targets = tracker.update(dets, frame.shape, frame.shape)
        del dets
        online_tlwhs = []
        online_ids = []
        online_scores = []

        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > min_box_area and not vertical:
                online_tlwhs.append(tlwh)
            
                online_scores.append(t.score)

                xmin, ymin, xmax, ymax = int(tlwh[0]),int(tlwh[1]),int(tlwh[0]+tlwh[2]),int(tlwh[1]+tlwh[3])

                if xmin > 0 and ymin>0 and ymax<frame.shape[0] and xmax<frame.shape[1]:
                    online_ids.append(tid)                
                    face = frame[int(ymin):int(ymax),int(xmin):int(xmax),:]

                    face = transforms.ToPILImage()(face)
                    if super_res:
                        face = transforms.Resize((56, 56),transforms.InterpolationMode.BICUBIC)(face)
                    else:
                        face = transforms.Resize((224, 224), transforms.InterpolationMode.BICUBIC)(face)
                    face = transforms.ToTensor()(face).unsqueeze(0)
                    faces.append(face)
                    bboxes.append(tlwh)

                    # cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),(0,255,0),1)
                    # cv2.putText(frame,str(tid),(int(tlwh[0]+tlwh[2]//2),int(tlwh[1]+tlwh[3]//2)),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0),1)

        if faces:
            faces_batch = torch.cat(faces,dim=0).to(device)
            if super_res:
                # SwinIR
                with torch.no_grad():
                    _, _, h_old, w_old = faces_batch.size()
                    h_pad = (h_old // window_size + 1) * window_size - h_old
                    w_pad = (w_old // window_size + 1) * window_size - w_old
                    faces_batch = torch.cat([faces_batch, torch.flip(faces_batch, [2])], 2)[:, :, :h_old + h_pad, :]
                    faces_batch = torch.cat([faces_batch, torch.flip(faces_batch, [3])], 3)[:, :, :, :w_old + w_pad]
                    outputs = test(faces_batch, swinIR, window_size)
                    faces_batch = outputs[..., :h_old * scale, :w_old * scale]

            # Age and Gender
            with torch.no_grad():
                (logits,probabs), out_gender = coral_model(faces_batch)
                probabs = probabs.cpu()
                out_gender = out_gender.cpu()
                logits = logits.cpu()
                ages = proba_to_label(probabs).float() + 1
                ages.numpy()
                low_labels, high_labels = proba_to_range(probabs)
                low_labels.numpy()
                high_labels.numpy()

                preds_gender = torch.nn.Sigmoid()(out_gender.reshape(-1,)).numpy()
                genders = list(map(lambda g : "M" if g < 0.5 else "F", preds_gender))
                
            for i in range(len(online_ids)):
                age = int(ages[i])
                gender = genders[i]
                tid = online_ids[i]
                bbox = bboxes[i]
                age_min = int(low_labels[i])
                age_max = int(high_labels[i])

                if tid in data:
                    data[tid]['age'].append(age)
                    data[tid]['gender'].append(gender)
                    data[tid]['range'].append([age_min,age_max])
                else:
                    data[tid] = {'age':[age],'gender':[gender],'age_min':[age_min],'age_max':[age_max]}

                # average age and gender
                avg_age = np.mean(data[tid]['age'])
                avg_gender = mode(data[tid]['gender'])
                avg_age_min = np.mean(data[tid]['age_min'])
                avg_age_max = np.mean(data[tid]['age_max'])

                xmin, ymin, xmax, ymax = int(bbox[0]),int(bbox[1]),int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3])

                label = avg_gender+" "+str(int(avg_age))#+" ("+str(avg_age_min)+"-"+str(avg_age_max)+")"
                row = [frame_id, online_ids[i], xmin, ymin, xmax-xmin, ymax-ymin, int(avg_age_min), int(avg_age_max), int(avg_age), avg_gender]
                csv_data.append(row)
                if avg_gender == 'M':
                    box_color = (255, 255, 0)
                    cv2.putText(frame, label, (int(xmin), int(ymin-3)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                else:
                    box_color = (191,0,255)
                    cv2.putText(frame, label, (int(xmin), int(ymin-3)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (191, 0, 255), 1)

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),box_color,2)
                cv2.putText(frame,str(tid),(int(tlwh[0]+tlwh[2]//2),int(tlwh[1]+tlwh[3]//2)),cv2.FONT_HERSHEY_SIMPLEX,0.4,box_color,2)
        
        frame_id+=1
        
        cv2.putText(frame, str(round(1/(time.time()-st), 2)), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # writing to csv
        csv_name = csv_folder+vid_name+".csv"
        if not os.path.isdir(csv_folder):
            os.mkdir(csv_folder)

        with open(csv_name, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(header)

            # write multiple rows
            writer.writerows(csv_data)
        if display:
            cv2.imshow("frame",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break   
        if save:
            vid_writer.write(frame)
    
    if save:
        vid_writer.release()
        print(f"Annotated video saved in {output_folder}\n")


    print(f"csv file saved in {csv_folder}\n")
    cap.release()
    cv2.destroyAllWindows() 


def img_pipeline(img_path, display, save):

    # loading image
    img = cv2.imread(img_path)

    yolov5, coral_model, peta_body, swinIR = load_models()

    # generate detections
    preds = yolov5(img, size=640, augment=False)
    detections = preds.xyxy[0]

    data = {}
    id_count = 0
    faces = []  # batch
    best_bodies = [] # batch
    map_body = []   # index mapping from face to best body
    heads = []  # all
    bodies = [] # all
    csv_data = []   # entire data to be printed on csv

    for det in detections:
        if det[5] == 1 and det[4] > 0.5:
            heads.append(det)
        elif det[5] == 0:
            bxmin, bymin, bxmax, bymax = int(det[0]),int(det[1]),int(det[2]),int(det[3])
            bodies.append(det)
    

    for head in heads:
        xmin, ymin, xmax, ymax, conf, _ = head
        head_box = (xmin, ymin, xmax, ymax)

        face = img[int(ymin):int(ymax),int(xmin):int(xmax),:]

        face = transforms.ToPILImage()(face)
        face = transforms.Resize((60,60),transforms.InterpolationMode.BICUBIC)(face)
        face = transforms.ToTensor()(face).unsqueeze(0)
        # face = transform_face(face)
        faces.append(face)

        max_iou = 0
        best_bod = 0
        best_diff = 1000
        found = False
        for body in bodies:
            bxmin, bymin, bxmax, bymax, conf, _ = body
            bod_box = (bxmin, bymin, bxmax, bymax)
            curr_iou = bb_intersection_over_union(bod_box, head_box)
            # curr_diff = (xmin + xmax - bxmin - bxmax)**2 + (ymin + ymax - bymin - bymax)**2 
            curr_diff = abs(ymin - bymin)
            if curr_iou >= max_iou or (curr_iou == max_iou and curr_diff < best_diff):
                max_iou = curr_iou
                best_bod = bod_box
                best_diff = curr_diff
            
        if max_iou > 0.3:
            found = True
        
        data[id_count] = {}
        data[id_count]['head'] = head_box

        if found:
            data[id_count]['body'] = best_bod
            bxmin, bymin, bxmax, bymax = best_bod
            map_body.append(id_count)
            best_bod_img = img[int(bymin):int(bymax),int(bxmin):int(bxmax),:]
            best_bod_img = transforms.ToPILImage()(best_bod_img)
            best_bod_img = transform_body(best_bod_img).unsqueeze(0)
            best_bodies.append(best_bod_img)

        id_count+=1
    
    if faces:
        faces_batch = torch.cat(faces,dim=0)

        # SwinIR
        with torch.no_grad():
            _, _, h_old, w_old = faces_batch.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            faces_batch = torch.cat([faces_batch, torch.flip(faces_batch, [2])], 2)[:, :, :h_old + h_pad, :]
            faces_batch = torch.cat([faces_batch, torch.flip(faces_batch, [3])], 3)[:, :, :, :w_old + w_pad]
            outputs = test(faces_batch.to(device), swinIR, window_size)
            faces_batch_hr = outputs[..., :h_old * scale, :w_old * scale]
            del outputs

        # Resizing Batch for Age-Gender
        faces_transformed = torch.rand((len(faces_batch),3,224,224))
        for i in range(len(faces_batch_hr)):
            faces_transformed[i] = transform_face(faces_batch_hr[i])

        del faces_batch_hr
        del faces_batch
        faces_transformed = faces_transformed.to(device)

        bodies_batch = torch.cat(best_bodies, dim=0).to(device)
        # Age and Gender Detection 
        with torch.no_grad():
            (logits,probabs), out_gender = coral_model(faces_transformed)
            probabs = probabs.cpu()
            out_gender = out_gender.cpu()
            logits = logits.cpu()
            ages = proba_to_label(probabs).float() + 1
            ages.numpy()
            low_labels, high_labels = proba_to_range(probabs)
            low_labels.numpy()
            high_labels.numpy()

            preds_gender = torch.nn.Sigmoid()(out_gender.reshape(-1,)).numpy()
            genders = list(map(lambda g : "M" if g < 0.5 else "F", preds_gender))

            # from body
            gen_body = peta_body(bodies_batch)
            gen_body = torch.nn.Softmax(dim = 1)(gen_body)
            gen_body = gen_body.cpu()
            preds_bod_gender = gen_body[:, 0]
            bod_genders = list(map(lambda g : "M" if g < 0.5 else "F", preds_bod_gender))


        # Annotating Image
        for id, person in data.items():
            age = int(ages[id])
            face_gender = genders[id]
            body_gender = 'error'
            if id in map_body:
                body_gender = bod_genders[map_body.index(id)]
            age_min = int(low_labels[id])
            age_max = int(high_labels[id])
            head_box = person['head']

            label = ''
            if id in map_body:
                label = face_gender + " " + str(int(age)) + " (" + str(age_min) + "-" + str(age_max)+")"
            else:
                label = face_gender + " " + str(int(age)) + " (" + str(age_min) + "-" + str(age_max)+")"

            # drawing head
            xmin, ymin, xmax, ymax = int(head_box[0]),int(head_box[1]),int(head_box[2]),int(head_box[3])
            # cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0,0,255), 1)

        
            # drawing body and putting label
            if 'body' in person.keys():
                bod_box = person['body']
                bxmin, bymin, bxmax, bymax = int(bod_box[0]),int(bod_box[1]),int(bod_box[2]),int(bod_box[3])
                
                if face_gender == 'M':
                    box_color = (255, 255, 0)
                    cv2.putText(img, label, (int(xmin), int(ymin-2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                else:
                    box_color = (191, 0, 255)
                    cv2.putText(img, label, (int(xmin), int(ymin-2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (191, 0, 255), 1)

                cv2.rectangle(img, (bxmin,bymin), (bxmax,bymax), box_color, 2)
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax),box_color,2)
                cv2.putText(img,str(id),(int(xmin + (xmax-xmin)//2),int(ymin+(ymax-ymin)//2)),cv2.FONT_HERSHEY_SIMPLEX,0.4,box_color,2)

                row = [0, id, xmin, ymin, ymax-ymin, xmax-xmin, age_min, age_max, age, body_gender]

            else:            
                # cv2.putText(img, label, (int(xmin), int(ymin-2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                if face_gender == 'M':
                    box_color = (255, 255, 0)
                    cv2.putText(img, label, (int(xmin), int(ymin-2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                else:
                    box_color = (191, 0, 255)
                    cv2.putText(img, label, (int(xmin), int(ymin-3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (191, 0, 255), 1)
                row = [0, id, xmin, ymin, ymax-ymin, xmax-xmin, age_min, age_max, age, face_gender]
                
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax),box_color,2)
                cv2.putText(img,str(id),(int(xmin + xmax-xmin)//2,int(ymin+(ymax-ymin)//2)),cv2.FONT_HERSHEY_SIMPLEX,0.4,box_color,2)

            csv_data.append(row)
        
        img_name = img_path.split('/')[-1].split('.')[-2]

        # writing to csv
        csv_folder = "csv_outputs/"
        csv_name = csv_folder+img_name+".csv"
        if not os.path.isdir(csv_folder):
            os.mkdir(csv_folder)

        with open(csv_name, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(header)

            # write multiple rows
            writer.writerows(csv_data)
        print(f"csv file saved in {csv_folder}\n")

        # saving annotated images
        if save:
            output_folder = 'outputs/'
            if not os.path.isdir(output_folder):
                os.mkdir(output_folder)
            cv2.imwrite(output_folder+img_name+'.jpg', img)
            print(f"Annotated image saved in {output_folder}\n")
        
    # displaying images
    if display:
        cv2.imshow("face",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run():
    parser = argparse.ArgumentParser(description='Pipeline for inference (BOSCH-Midprep')
    parser.add_argument('path', type=str,
                        help='image or video name')
    parser.add_argument('--display', action='store_true',
                        help='show image/video output in openCV window')
    parser.add_argument('--save', action='store_true',
                        help='saves image in ./output')
    parser.add_argument('--hrvid', action='store_true',
                        help='high res mode for vid (low fps)')
    
    args = parser.parse_args()
    
    if args.path.split('.')[-1] in img_type:
        img_pipeline(args.path, args.display, args.save)
    elif args.path.split('.')[-1] in vid_type:
        vid_pipeline(args.path, args.display, args.save, args.hrvid)
    else:
        print('Check input path')

if __name__ == '__main__':
    run()