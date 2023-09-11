from medpy import metric
import numpy as np

def get_metrics(pred, gt, voxelspacing=(0.5, 0.5, 0.5)):
    r"""
    Get statistic metrics of segmentation

    These metrics include: Dice, Jaccard, Hausdorff Distance, 95% Hausdorff Distance,
    and Average surface distance(ASD) metric.

    If the prediction result is 0s, we set hd, hd95, asd 10.0 to avoid errors.

    Parameters:
    -----------
    pred: 3D numpy ndarray
        binary prediction results

    gt: 3D numpy ndarray
        binary ground truth

    voxelspacing: tuple of 3 floats. default: (0.5, 0.5, 0.5)
        voxel space of 3D image

    Returns:
    --------
    metrics: dict of 5 metrics
        dict{dsc, jc, hd, hd95, asd}
    """

    dsc = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    vs = np.sum(gt) * voxelspacing[0] * voxelspacing[1] * voxelspacing[2]
    precision = metric.binary.precision(pred, gt)
    recall = metric.binary.recall(pred, gt)
    rvd=metric.binary.ravd(pred, gt)


    if np.sum(pred) == 0:
        print('=> prediction is 0s! ')
        hd = 32.
        hd95 = 32.
        asd = 10.
    else:
        hd = metric.binary.hd(pred, gt, voxelspacing=voxelspacing)
        hd95 = metric.binary.hd95(pred, gt, voxelspacing=voxelspacing)
        asd = metric.binary.asd(pred, gt, voxelspacing=voxelspacing)
        assd = metric.binary.assd(pred, gt, voxelspacing=voxelspacing)
        #
        sds1=metric.binary.__surface_distances(pred, gt, voxelspacing=voxelspacing)
        sds2 = metric.binary.__surface_distances(gt, pred, voxelspacing=voxelspacing)
        sd11 = sds1 * sds1
        sd22 = sds2 * sds2
        rmsd = np.sqrt(sd11.sum() + sd22.sum()) / np.sqrt(sds1.size + sds2.size)
        #

    metrics = {'dsc': dsc, 'jc': jc, 'hd': hd, 'hd95': hd95, 'asd': asd,
                'precision':precision, 'recall':recall, 'vs': vs,'assd':assd,'rvd':rvd,'rmsd':rmsd}
    return metrics

def get_metrics_2d(pred, gt, voxelspacing=(0.21, 0.21)):
    r"""
    Get statistic metrics of segmentation

    These metrics include: Dice, Jaccard, Hausdorff Distance, 95% Hausdorff Distance,
    , Pixel Wise Accuracy, Precision and Recall.

    If the prediction result is 0s, we set hd, hd95, 10.0 to avoid errors.

    Parameters:
    -----------

        pred: 2D numpy ndarray
            binary prediction results

        gt: 2D numpy ndarray
            binary ground truth

        voxelspacing: tuple of 2 floats. default: (0.21, 0.21)
            voxel space of 2D image

    Returns:
    --------

        metrics: dict of 7 metrics
            dict{dsc, jc, hd, hd95, precision, recall, acc}

    """

    dsc = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    precision = metric.binary.precision(pred, gt)
    recall = metric.binary.recall(pred, gt)

    acc = (pred == gt).sum() / len(gt.flatten())

    if np.sum(pred) == 0:
        #print('=> prediction is 0s! ')
        hd = 10
        hd95 = 10
    else:
        hd = metric.binary.hd(pred, gt, voxelspacing=voxelspacing)
        hd95 = metric.binary.hd95(pred, gt, voxelspacing=voxelspacing)

    metrics = {'dsc': dsc, 'jc': jc, 'hd': hd, 'hd95': hd95,
                'precision':precision, 'recall':recall, 'acc':acc}
    return metrics

def get_dice(pred, gt):
    dice = metric.binary.dc(pred, gt)

    return dice

from skimage import measure
def connected_component(image):
    # 标记输入的3D图像
    label, num = measure.label(image, connectivity=2, return_num=True)
    if num < 1:
        return image

    # 获取对应的region对象
    region = measure.regionprops(label)
    # 获取每一块区域面积并排序
    num_list = [i for i in range(1, num + 1)]
    area_list = [region[i - 1].area for i in num_list]
    num_list_sorted = sorted(num_list, key=lambda x: area_list[x - 1])[::-1]
    # 去除面积较小的连通域
    if len(num_list_sorted) > 1:
        # for i in range(3, len(num_list_sorted)):
        for i in num_list_sorted[1:]:
            # label[label==i] = 0
            label[region[i - 1].slice][region[i - 1].image] = 0
        num_list_sorted = num_list_sorted[:1]
    label[label>0]=1
    return label
