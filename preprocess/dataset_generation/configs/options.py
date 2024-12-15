
def get_dataset(opt):

    print("Loading dataset %s ..." % opt.dataset)
    if opt.dataset == "hm3d":
        
        # basedir="./dataset_generation/"
        basedir="/wudang_vuc_3dc_afs/chenzheng/dataset/"

        opt.train_data_path = (
            basedir
            + "pointnav/hm3d/train/train.json.gz"
        )
        opt.val_data_path = (
            basedir
            + "pointnav/hm3d/test/test.json.gz"
        )
        opt.test_data_path = (
            basedir
            + "pointnav/hm3d/val/val.json.gz"
        )
        opt.scenes_dir = basedir # this should store hm3d
    elif opt.dataset == "replica":
        # replace by synsin json.

        # these are not used.
        opt.train_data_path = (
            "/private/home/ow045820/projects/habitat/habitat-api/"
            + "data/datasets/pointnav/replica/v1/train/train.json.gz"
        )
        opt.val_data_path = (
            "/private/home/ow045820/projects/habitat/habitat-api/"
            + "data/datasets/pointnav/replica/v1/val/val.json.gz"
        )
        opt.test_data_path = (
            "/private/home/ow045820/projects/habitat/habitat-api/"
            + "data/datasets/pointnav/replica/v1/test/test.json.gz"
        )

        # used
        basedir="/wudang_vuc_3dc_afs/chenzheng"
        opt.scenes_dir = basedir #"/checkpoint/ow045820/data/replica/"
    else:
        raise NotImplementedError
    from dataset_generation.configs.habitat_data import HabitatImageGenerator as Dataset
    return Dataset # unused
