import bpu_infer_lib
import numpy as np

inf = bpu_infer_lib.Infer(False)
inf.load_model("lpr.bin")

input_data = np.fromfile("test.bin", dtype=np.float32).reshape(1,3,24,94)

inf.read_input(input_data, 0)
inf.forward(more=True)
inf.get_output()

res = inf.outputs[0].data
res = res.reshape(1,68,18)

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]
def reprocess(pred):
    pred_data = pred[0]
    pred_label = np.argmax(pred_data, axis=0)
    no_repeat_blank_label = []
    pre_c = pred_label[0]
    if pre_c != len(CHARS) - 1: 
        no_repeat_blank_label.append(pre_c)
    for c in pred_label: 
        if (pre_c == c) or (c == len(CHARS) - 1):
            if c == len(CHARS) - 1:
                pre_c = c
            continue
        no_repeat_blank_label.append(c)
        pre_c = c
    char_list = [CHARS[i] for i in no_repeat_blank_label]
    return ''.join(char_list)

plate_str = reprocess(res)
print(plate_str) 
