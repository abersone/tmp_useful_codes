import os
import numpy as np
import matplotlib.pyplot as plt

def plot_precision_recall_vs_confidence(recall, precision, confidence):
    plt.figure(figsize=(10, 5))

    # Plot precision and recall against confidence
    plt.subplot(1, 2, 1)
    plt.plot(confidence, precision, label='Precision')
    plt.plot(confidence, recall, label='Recall')
    plt.xlabel('Confidence')
    plt.ylabel('Score')
    plt.title('Precision and Recall vs Confidence')
    plt.legend()

    # Plot precision against recall
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label='Precision vs Recall')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs Recall')
    plt.legend()

    plt.tight_layout()
    plt.show()

def calculate_iou(box1, boxes2):
    iou_scores = np.zeros(len(boxes2))
    for j in range(len(boxes2)):
        x1 = max(box1[0], boxes2[j, 0])
        y1 = max(box1[1], boxes2[j, 1])
        x2 = min(box1[0] + box1[2], boxes2[j, 0] + boxes2[j, 2])
        y2 = min(box1[1] + box1[3], boxes2[j, 1] + boxes2[j, 3])
       
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = box1[2] * box1[3]
        area2 = boxes2[j, 2] * boxes2[j, 3]
        union = area1 + area2 - intersection

        iou_scores[j] = intersection / union

    return iou_scores

def compute_mAP_kernel(det_boxes_list, labels_num, thres_conf):
    num_classes = 1  # 假设有2个类别（0和1），可以根据您的实际情况调整
    all_true_positives = [[] for _ in range(num_classes)]
    all_false_positives = [[] for _ in range(num_classes)]
    all_confidences = [[] for _ in range(num_classes)]
    
    for i in range(num_classes):
        det_boxes_list = np.array(det_boxes_list)  # 将det_boxes转换为NumPy数组
        class_det_boxes = det_boxes_list[det_boxes_list[:, 1] == i]  # 选择特定类别的检测框
        true_positives = class_det_boxes[:, 2]
        false_positives = class_det_boxes[:, 3]
        confidences = class_det_boxes[:, 4]
        
        all_true_positives[i].extend(true_positives)
        all_false_positives[i].extend(false_positives)
        all_confidences[i].extend(confidences)
            
    
    average_precisions = []
    
    for i in range(num_classes):
        true_positives = np.array(all_true_positives[i])
        false_positives = np.array(all_false_positives[i])
        confidences = np.array(all_confidences[i])
        #print('true_positives:', true_positives)
        #print('false_positives:', false_positives)
        #print('confidences:', confidences)
        label_num = labels_num[i]
        #print('label_num:', label_num)
        
        sorted_indices = np.argsort(-confidences)
        sorted_true_positives = true_positives[sorted_indices]
        sorted_false_positives = false_positives[sorted_indices]
        sorted_confidences = confidences[sorted_indices]
        
        print('len(sorted_true_positives):', len(sorted_true_positives))
        #print('sorted_true_positives:', sorted_true_positives)
        #print('sorted_false_positives:', sorted_false_positives)
        #print('sorted_confidences:', sorted_confidences)
        
        mask = sorted_confidences >= thres_conf
        filtered_true_positives = sorted_true_positives[mask]
        filtered_false_positives = sorted_false_positives[mask]
        filtered_confidences = sorted_confidences[mask]
        print('len(filtered_true_positives):', len(filtered_true_positives))
        #print('filtered_true_positives:', filtered_true_positives)
        #print('filtered_false_positives:', filtered_false_positives)
        
        cumulative_true_positives = np.cumsum(filtered_true_positives)
        cumulative_false_positives = np.cumsum(filtered_false_positives)
        #print('len(cumulative_true_positives):', len(cumulative_true_positives))
        #print('cumulative_true_positives:',cumulative_true_positives)
        #print('cumulative_false_positives:',cumulative_false_positives)
        
        recall = cumulative_true_positives / labels_num
        precision = cumulative_true_positives / (cumulative_true_positives + cumulative_false_positives)
        #print('recall', recall)
        #print('precision', precision)
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))
        #print('mrec', mrec)
        #print('mpre', mpre)
        average_precision = recall[0] * precision[0]
        for i in range(1, len(recall)):
            average_precision += (recall[i] - recall[i-1]) * precision[i]
            
        # print('average_precision:', average_precision)
        average_precisions.append(average_precision)
        
        # Compute the precision envelope
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

        # Integrate area under curve by yolov5
        method = 'continuous'  # methods: 'continuous'(the same with self R&D), 'interp'
        if method == 'interp':
            x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
            ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
        else:  # 'continuous'
            i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve
        print('yolo_average_precision:', ap)
    mean_average_precision = np.mean(average_precisions)
    
    return mean_average_precision, recall, precision, filtered_confidences

def compute_mAP(predictions_folder, labels_folder, num_classes, confidence_threshold, do_normalize=True):
    ap_scores = []
    labels_num = []
    null_num = 0
    for class_id in range(num_classes):
        # pred_result: [pred_class, true_class, TP, FP, conf] 1 x 5
        pred_result = []
        label_num = 0
        for filename in os.listdir(labels_folder):
            if not filename.endswith('txt'):
                continue

            img_idx = filename.split('.')[0]
            true_labels_file = os.path.join(labels_folder, filename)
            pred_labels_file = os.path.join(predictions_folder, filename)
            #print("true_labels_file:", true_labels_file)
            #print("pred_labels_file:", pred_labels_file)
            with open(true_labels_file, 'r') as true_file, open(pred_labels_file, 'r') as pred_file:
                true_labels_data = np.atleast_2d(np.loadtxt(true_file, dtype=float))  # Ensure at least 2D array
                pred_labels_data = np.atleast_2d(np.loadtxt(pred_file, dtype=float))  # Ensure at least 2D array
            if pred_labels_data.size == 0:
                null_num = null_num + 1
                
            label_num = label_num + len(true_labels_data)
            #print(pred_labels_data.shape)
            if pred_labels_data.shape[0] == 0 or pred_labels_data.shape[1] == 0:
                continue
                
            if true_labels_data.shape[0] == 1:
                true_labels_data = np.array(true_labels_data)  # Ensure 2D array for single-row case
            if pred_labels_data.shape[0] == 1:
                pred_labels_data = np.array(pred_labels_data)  # Ensure 2D array for single-row case
            
            # for yolo label
            ## normal format
            if do_normalize:
                image_size = pred_labels_data[0, 6:8]  # Get image size from the first prediction
                pred_labels_data[:, 1] /= image_size[0]  # Normalize x
                pred_labels_data[:, 2] /= image_size[1]  # Normalize y
                pred_labels_data[:, 3] /= image_size[0]  # Normalize width
                pred_labels_data[:, 4] /= image_size[1]  # Normalize height
            ## pred yolo format
            else:
                pred_labels_data[:, 1] -= pred_labels_data[:, 3] / 2  # Convert center_x to x
                pred_labels_data[:, 2] -= pred_labels_data[:, 4] / 2  # Convert center_y to y
            
            true_labels_data[:, 1] -= true_labels_data[:, 3] / 2  # Convert center_x to x
            true_labels_data[:, 2] -= true_labels_data[:, 4] / 2  # Convert center_y to y
            
            # Match predictions with labels based on IOU
            matched = [False] * len(true_labels_data)
            for pred_box in pred_labels_data:
                iou_scores = calculate_iou(pred_box[1:5], true_labels_data[:, 1:5])
                #print(iou_scores)
                best_match_id = np.argmax(iou_scores)
                iou_threshold = 0.5  # Adjust this threshold as needed

                if iou_scores[best_match_id] > iou_threshold:
                    if not matched[best_match_id]:
                        matched[best_match_id] = True
                        pred_result.append([pred_box[0], class_id, 1, 0, pred_box[5]]) 
                    else:
                        pred_result.append([pred_box[0], class_id, 0, 1, pred_box[5]])
                else:
                    # Consider this as a false positive
                    pred_result.append([pred_box[0], class_id, 0, 1, pred_box[5]])
        
        #print("pred_result:", len(pred_result))
        labels_num.append(label_num)
        
    mAP = []
    mAP, recall, precision, filtered_confidences = compute_mAP_kernel(pred_result, labels_num, confidence_threshold)
    return mAP, recall, precision, filtered_confidences, null_num
    
if __name__ == '__main__':
    predictions_folder = 'align_score/20241220/label_dnn'
    labels_folder = 'align_score/20241220/label_gt'
    num_classes = 1
    confidence_threshold = 0.7
    do_normalize=True # True means inference by dnn with onnx; False means inference by ultralytics with pt model
    mAP, recall, precision, filtered_confidences, null_num = compute_mAP(predictions_folder, labels_folder, num_classes, confidence_threshold, do_normalize=do_normalize)
    plot_precision_recall_vs_confidence(recall, precision, filtered_confidences)
    print("mAP: {:.2f}%".format(mAP*100.0))
    print("null txt numbers:", null_num)