from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import cv2

def Standard_colors():

    STANDARD_COLORS = [
        'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
        'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
        'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
        'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
        'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
        'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
        'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
        'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
        'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
        'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
        'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
        'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
        'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
        'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
        'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
        'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
        'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
        'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
        'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
        'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
        'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
        'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
        'WhiteSmoke', 'Yellow', 'YellowGreen'
    ]
    return STANDARD_COLORS


def Load_label(path):
    PATH_TO_LABELS = path
    return label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


def WriteResult(nameTxt, DataMemory,category_index):
  print(nameTxt)
  f = open(nameTxt, "w")
  for i in range(0,len(DataMemory)):
    f.write(str(DataMemory)+"\n")
  f.write("\n")
  for i in range(0,len(DataMemory)):
    st=str(category_index[DataMemory[i][1][0]]['name'])+" "+str(DataMemory[i][0])+" has an average speed of "+str(DataMemory[i][4][2])+" calculated on "+str(DataMemory[i][6][0]) +"frames \n"
    f.write(str(st))
  f.close()


'''
import collections
import six
import numpy as np
import PIL.Image as Image
def DrawBoxesUtils(

    image,
    boxes,
    classes,
    scores,
    category_index,
    center,
    instance_masks=None,
    instance_boundaries=None,
    keypoints=None,
    keypoint_scores=None,
    keypoint_edges=None,
    track_ids=None,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=.5,
    line_thickness=4,
    mask_alpha=.4,
    groundtruth_box_visualization_color='black'):
 
 #Definisco di default le strutture dati che riempirò nella funzione
  box_to_display_str_map = collections.defaultdict(list)
  box_to_color_map = collections.defaultdict(str)
  box_to_instance_masks_map = {}
  box_to_instance_boundaries_map = {}
  box_to_keypoints_map = collections.defaultdict(list)
  box_to_keypoint_scores_map = collections.defaultdict(list)
  box_to_track_ids_map = {}

  final_label=[]
  counterOfElements=[]

  #for i in range(boxes.shape[0]):



  if not max_boxes_to_draw:
    max_boxes_to_draw = boxes.shape[0]
  for i in range(boxes.shape[0]):
    if max_boxes_to_draw == len(box_to_color_map):
      break
    if scores is None or scores[i] > min_score_thresh:
      box = tuple(boxes[i].tolist())
      if instance_masks is not None:
        box_to_instance_masks_map[box] = instance_masks[i]
      if instance_boundaries is not None:
        box_to_instance_boundaries_map[box] = instance_boundaries[i]
      if keypoints is not None:
        box_to_keypoints_map[box].extend(keypoints[i])
      if keypoint_scores is not None:
        box_to_keypoint_scores_map[box].extend(keypoint_scores[i])
      if track_ids is not None:
        box_to_track_ids_map[box] = track_ids[i]
      if scores is None:
        box_to_color_map[box] = groundtruth_box_visualization_color
      else:
        display_str = ''
        if classes[i] in six.viewkeys(category_index):
          class_name = category_index[classes[i]]['name']
        else:
          class_name = 'N/A'
        display_str = str(class_name)

        ### MODIFICA PER IL CONTATORE DI LABEL 
        if display_str in final_label:
          counterOfElements[final_label.index(display_str)]+=1
        else:
          final_label.append(display_str)
          counterOfElements.append(1)

        
        STANDARD_COLORS=Standard_colors()


        box_to_display_str_map[box].append(display_str)
        if track_ids is not None:
          prime_multipler = viz_utils._get_multiplier_for_color_randomness()
          box_to_color_map[box] = STANDARD_COLORS[
              (prime_multipler * track_ids[i]) % len(STANDARD_COLORS)]
        else:
          box_to_color_map[box] = STANDARD_COLORS[
              classes[i] % len(STANDARD_COLORS)]

  # Draw all boxes onto image.
  for box, color in box_to_color_map.items():
    ymin, xmin, ymax, xmax = box
    print("Box to display's type  ",type(box_to_display_str_map[box]))
    print("Box to display  ",box_to_display_str_map[box])
    print("Color",color)
    print("")
    viz_utils.draw_bounding_box_on_image_array(image, ymin, xmin, ymax , xmax, color=color, thickness=0, display_str_list=box_to_display_str_map[box], use_normalized_coordinates=True)

  print("---------------------------------")

  return final_label,counterOfElements, image
'''