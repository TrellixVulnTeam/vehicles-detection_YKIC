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

