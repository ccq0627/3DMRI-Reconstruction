import numpy as np
import nibabel as nib
import os.path as osp
from argparse import ArgumentParser
import sys

sys.path.append("./")
from r2_gaussian.utils.argument_utils import ParamGroup

class InitializeParams(ParamGroup):
    pass
    # def __init__(self, parser,):
    #     self.
    #     super().__init__(parser, "Initialization Parameters")


def main(args):
    data_path = args.path
    if args.output == None:
        dir_path = osp.dirname(data_path)
    else:
        dir_path = args.output
    save_path = osp.join(dir_path, "vol_gt" + ".npy")
    if not osp.exists(save_path):
        data_5d = nib.ni1.load(data_path).get_fdata()
        data = data_5d[0, 0, :, :, :]
        vol_gt = data / np.max(data)

        np.save(save_path, vol_gt)

    if False:
        import pyvista as pv
        vol_gt = np.load(save_path)
        plotter = pv.Plotter(window_size=(800,800), line_smoothing=True, off_screen=False)
        plotter.add_volume(vol_gt)
        plotter.show_axes()
        plotter.show()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to MRI data", default="MRIdata/00000.nii.gz")
    parser.add_argument("--output", type=str, help="Output folder", default=None)

    args = parser.parse_args()

    main(args)