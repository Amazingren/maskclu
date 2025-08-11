import random

import numpy as np
import torch


def pc_normalize_np(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud datasets, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def angle_axis(angle, axis):
    # type: (float, np.ndarray) -> float
    r"""Returns a 4x4 rotation matrix that performs a rotation around axis by angle
    Parameters
    ----------
    angle : float
        Angle to rotate by
    axis: np.ndarray
        Axis to rotate about
    Returns
    -------
    torch.Tensor
        3x3 rotation matrix
    """
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)

    # yapf: disable
    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                               [u[2], 0.0, -u[0]],
                               [-u[1], u[0], 0.0]])

    R = torch.from_numpy(
        cosval * np.eye(3)
        + sinval * cross_prod_mat
        + (1.0 - cosval) * np.outer(u, u)
    )
    # yapf: enable
    return R.float()


def fps(points, num):
    cids = []
    cid = np.random.choice(points.shape[0])
    cids.append(cid)
    id_flag = np.zeros(points.shape[0])
    id_flag[cid] = 1

    dist = torch.zeros(points.shape[0]) + 1e4
    dist = dist.type_as(points)
    while np.sum(id_flag) < num:
        dist_c = torch.norm(points - points[cids[-1]], p=2, dim=1)
        dist = torch.where(dist < dist_c, dist, dist_c)
        dist[id_flag == 1] = 1e4
        new_cid = torch.argmin(dist)
        id_flag[new_cid] = 1
        cids.append(new_cid)
    cids = torch.Tensor(cids)
    return cids


class PointcloudRotate(object):
    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, 0, sinval],
                                        [0, 1, 0],
                                        [-sinval, 0, cosval]])
            R = torch.from_numpy(rotation_matrix.astype(np.float32)).to(pc.device)
            pc[i, :, :] = torch.matmul(pc[i], R)
        return pc


class PointcloudScaleAndTranslate(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])

            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda()) + torch.from_numpy(
                xyz2).float().cuda()

        return pc


class PointcloudJitter(object):
    def __init__(self, std=0.01, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            jittered_data = pc.new(pc.size(1), 3).normal_(
                mean=0.0, std=self.std
            ).clamp_(-self.clip, self.clip)
            pc[i, :, 0:3] += jittered_data

        return pc


class PointcloudScale(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2.):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])

            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda())

        return pc


class PointcloudTranslate(object):
    def __init__(self, translate_range=0.2):
        self.translate_range = translate_range

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])

            pc[i, :, 0:3] = pc[i, :, 0:3] + torch.from_numpy(xyz2).float().cuda()

        return pc


class PointcloudRandomInputDropout(object):
    def __init__(self, max_dropout_ratio=0.5):
        assert max_dropout_ratio >= 0 and max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            dropout_ratio = np.random.random() * self.max_dropout_ratio  # 0~0.875
            drop_idx = np.where(np.random.random((pc.size()[1])) <= dropout_ratio)[0]
            if len(drop_idx) > 0:
                cur_pc = pc[i, :, :]
                cur_pc[drop_idx.tolist(), 0:3] = cur_pc[0, 0:3].repeat(len(drop_idx), 1)  # set to the first point
                pc[i, :, :] = cur_pc

        return pc


class RandomHorizontalFlip(object):

    def __init__(self, upright_axis='z', is_temporal=False):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.is_temporal = is_temporal
        self.D = 4 if is_temporal else 3
        self.upright_axis = {'x': 0, 'y': 1, 'z': 2}[upright_axis.lower()]
        # Use the rest of axes for flipping.
        self.horz_axes = set(range(self.D)) - set([self.upright_axis])

    def __call__(self, coords):
        bsize = coords.size()[0]
        for i in range(bsize):
            if random.random() < 0.95:
                for curr_ax in self.horz_axes:
                    if random.random() < 0.5:
                        coord_max = torch.max(coords[i, :, curr_ax])
                        coords[i, :, curr_ax] = coord_max - coords[i, :, curr_ax]
        return coords


class PointcloudToTensor(object):
    def __call__(self, points):
        return torch.from_numpy(points).float()


class PointcloudSample(object):
    def __init__(self, num_pt=4096):
        self.num_points = num_pt

    def __call__(self, points):
        pc = points.numpy()
        # pt_idxs = np.arange(0, self.num_points)
        pt_idxs = np.arange(0, points.shape[0])
        np.random.shuffle(pt_idxs)
        pc = pc[pt_idxs[0:self.num_points], :]
        return torch.from_numpy(pc).float()


def pc_normalize(tensor, zero_mean=True):
    # tensor shape: [B, N, D] where B is the batch size, N is the number of points, and D is the dimensionality (usually 3)
    if zero_mean:
        # Calculate mean along the points dimension (dim=1) and keep batch and feature dimensions
        m = tensor.mean(dim=1, keepdim=True)  # [B, N, D] -> [B, 1, D]
        v = tensor - m
    else:
        v = tensor

    # Calculate the L2 norm across the feature dimension for each point in the batch
    nn = v.norm(p=2, dim=2)  # [B, N, D] -> [B, N]
    # Find the max norm value for each batch to normalize within each batch
    nmax = nn.max(dim=1, keepdim=True)[0]  # [B, N] -> [B, 1]

    # Normalize each point cloud in the batch by dividing by the max norm value
    return v / nmax.unsqueeze(-1)  # [B, N, D]


class PointcloudNormalize(object):
    def __init__(self, radius=1):
        self.radius = radius

    def __call__(self, batch_points):
        # Expecting batch_points to be of shape [B, N, 3] where B is batch size, N is number of points, and 3 is the dimensionality
        batch_points[:, :, 0:3] = pc_normalize(batch_points[:, :, 0:3])
        return batch_points



class PointcloudRemoveInvalid(object):
    def __init__(self, invalid_value=0):
        self.invalid_value = invalid_value

    def __call__(self, points):
        pc = points.numpy()
        valid = np.sum(pc, axis=1) != self.invalid_value
        pc = pc[valid, :]
        return torch.from_numpy(pc).float()


class PointcloudRandomCrop(object):
    def __init__(self, x_min=0.6, x_max=1.1, ar_min=0.75, ar_max=1.33, p=1,
                 min_num_points=4096, max_try_num=10):
        self.x_min = x_min
        self.x_max = x_max

        self.ar_min = ar_min
        self.ar_max = ar_max

        self.p = p

        self.max_try_num = max_try_num
        self.min_num_points = min_num_points

    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points
        points = points.numpy()

        isvalid = False
        try_num = 0
        while not isvalid:
            coord_min = np.min(points[:, :3], axis=0)
            coord_max = np.max(points[:, :3], axis=0)
            coord_diff = coord_max - coord_min
            # resampling later, so only consider crop here
            new_coord_range = np.zeros(3)
            new_coord_range[0] = np.random.uniform(self.x_min, self.x_max)
            ar = np.random.uniform(self.ar_min, self.ar_max)
            # new_coord_range[1] = np.random.uniform(self.ar_min, self.ar_max) * new_coord_range[0]
            # new_coord_range[2] = np.random.uniform(self.ar_min, self.ar_max) * new_coord_range[0]
            new_coord_range[1] = new_coord_range[0] * ar
            new_coord_range[2] = new_coord_range[0] / ar
            # new_coord_range = np.where(new_coord_range>1, 1, new_coord_range)

            new_coord_min = np.random.uniform(0, 1 - new_coord_range)
            new_coord_max = new_coord_min + new_coord_range

            new_coord_min = coord_min + coord_diff * new_coord_min
            new_coord_max = coord_min + coord_diff * new_coord_max

            new_indices = (points[:, :3] > new_coord_min) & (points[:, :3] < new_coord_max)
            new_indices = np.sum(new_indices, axis=1) == 3
            new_points = points[new_indices]

            # other_num = points.shape[0] - new_points.shape[0]
            # if new_points.shape[0] > 0:
            #     isvalid = True
            if self.min_num_points <= new_points.shape[0] < points.shape[0]:
                isvalid = True

            try_num += 1
            if try_num > self.max_try_num:
                return torch.from_numpy(points).float()

        # other_indices = np.random.choice(np.arange(new_points.shape[0]), other_num)
        # other_points = new_points[other_indices]
        # new_points = np.concatenate([new_points, other_points], axis=0)
        return torch.from_numpy(new_points).float()


class PointcloudRandomCutout(object):
    def __init__(self, ratio_min=0.3, ratio_max=0.6, p=1, min_num_points=4096, max_try_num=10):
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
        self.p = p
        self.min_num_points = min_num_points
        self.max_try_num = max_try_num

    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points
        points = points.numpy()
        try_num = 0
        valid = False
        while not valid:
            coord_min = np.min(points[:, :3], axis=0)
            coord_max = np.max(points[:, :3], axis=0)
            coord_diff = coord_max - coord_min

            cut_ratio = np.random.uniform(self.ratio_min, self.ratio_max, 3)
            new_coord_min = np.random.uniform(0, 1 - cut_ratio)
            new_coord_max = new_coord_min + cut_ratio

            new_coord_min = coord_min + new_coord_min * coord_diff
            new_coord_max = coord_min + new_coord_max * coord_diff

            cut_indices = (points[:, :3] > new_coord_min) & (points[:, :3] < new_coord_max)
            cut_indices = np.sum(cut_indices, axis=1) == 3

            # print(np.sum(cut_indices))
            # other_indices = (points[:, :3] < new_coord_min) | (points[:, :3] > new_coord_max)
            # other_indices = np.sum(other_indices, axis=1) == 3
            try_num += 1

            if try_num > self.max_try_num:
                return torch.from_numpy(points).float()

            # cut the points, sampling later

            if points.shape[0] - np.sum(cut_indices) >= self.min_num_points and np.sum(cut_indices) > 0:
                # print (np.sum(cut_indices))
                points = points[cut_indices == False]
                valid = True

        # if np.sum(other_indices) > 0:
        #     comp_indices = np.random.choice(np.arange(np.sum(other_indices)), np.sum(cut_indices))
        #     points[cut_indices] = points[comp_indices]
        return torch.from_numpy(points).float()


def uniform_2_sphere(num: int):
    """Uniform sampling on a 2-sphere for batch processing.
    Args:
        num: Number of vectors to sample (for batch processing).
    Returns:
        Random vectors (torch.Tensor) of size (num, 3) with norm 1.
    """
    phi = torch.rand(num) * 2 * torch.pi  # Uniformly sample phi in [0, 2*pi]
    cos_theta = torch.rand(num) * 2 - 1   # Uniformly sample cos(theta) in [-1, 1]
    theta = torch.acos(cos_theta)
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)

    return torch.stack((x, y, z), dim=-1)  # Shape: (num, 3)


class PointcloudSphereCrop(object):
    """Randomly crops the *source* point cloud for a batch."""
    def __init__(self, p_keep=0.85, target_num_points=1024):
        self.p_keep = p_keep
        self.target_num_points = target_num_points  # Desired size after padding

    def crop(self, points, p_keep):
        bsize = points.shape[0]
        rand_directions = uniform_2_sphere(bsize).to(points.device)
        centroids = torch.mean(points[:, :, :3], dim=1)
        points_centered = points[:, :, :3] - centroids[:, None, :]

        dist_from_plane = torch.einsum('bij,bj->bi', points_centered, rand_directions)
        thresholds = torch.quantile(dist_from_plane, 1.0 - p_keep, dim=1, keepdim=True)
        masks = dist_from_plane > thresholds

        # Apply mask and use random sampling to pad each point cloud to target_num_points
        cropped_points = []
        for i in range(bsize):
            selected_points = points[i][masks[i]]
            num_selected = selected_points.shape[0]

            if num_selected >= self.target_num_points:
                # Randomly sample to reduce points to target_num_points
                indices = torch.randperm(num_selected)[:self.target_num_points]
                sampled_points = selected_points[indices]
            else:
                # Randomly sample points to fill up to target_num_points
                indices = torch.randint(0, num_selected, (self.target_num_points - num_selected,))
                sampled_points = torch.cat([selected_points, selected_points[indices]], dim=0)

            cropped_points.append(sampled_points)

        return torch.stack(cropped_points)

    def __call__(self, sample):
        if self.p_keep >= 1.0:
            return sample  # No cropping needed
        return self.crop(sample, self.p_keep)


class PointcloudUpSampling(object):
    def __init__(self, max_num_points, radius=0.1, nsample=5, centroid="random"):
        self.max_num_points = max_num_points
        # self.radius = radius
        self.centroid = centroid
        self.nsample = nsample

    def __call__(self, points):
        p_num = points.shape[0]
        if p_num > self.max_num_points:
            return points

        c_num = self.max_num_points - p_num

        if self.centroid == "random":
            cids = np.random.choice(np.arange(p_num), c_num)
        else:
            assert self.centroid == "fps"
            fps_num = c_num / self.nsample
            fps_ids = fps(points, fps_num)
            cids = np.random.choice(fps_ids, c_num)

        xyzs = points[:, :3]
        loc_matmul = torch.matmul(xyzs, xyzs.t())
        loc_norm = xyzs * xyzs
        r = torch.sum(loc_norm, -1, keepdim=True)
        r_t = r.t()
        dist = r - 2 * loc_matmul + r_t
        # adj_matrix = torch.sqrt(dist + 1e-6)

        dist = dist[cids]
        # adj_sort = torch.argsort(adj_matrix, 1)
        adj_topk = torch.topk(dist, k=self.nsample * 2, dim=1, largest=False)[1]

        uniform = np.random.uniform(0, 1, (cids.shape[0], self.nsample * 2))
        median = np.median(uniform, axis=1, keepdims=True)
        # choice = adj_sort[:, 0:self.nsample*2][uniform > median]  # (c_num, n_samples)
        choice = adj_topk[uniform > median]  # (c_num, n_samples)

        choice = choice.reshape(-1, self.nsample)

        sample_points = points[choice]  # (c_num, n_samples, 3)

        new_points = torch.mean(sample_points, dim=1)
        new_points = torch.cat([points, new_points], 0)

        return new_points


def points_sampler(points, num, fs=False):
    if points.shape[0] > num:
        if fs:
            return farthest_point_sample(points, num)
        pt_idxs = np.arange(0, points.shape[0])
        np.random.shuffle(pt_idxs)
        points = points[pt_idxs[0:num], :]
    return points


if __name__ == "__main__":
    points = np.random.uniform(0, 100, [1024, 3])
    points = torch.Tensor(points).cuda()
    # crop = PointcloudRandomCrop()
    # crop_points = crop(points)

    # cutout = PointcloudRandomCutout()
    # cutout_points = cutout(points)
    # print(np.max(points,axis=0), np.min(points,axis=0),
    # torch.max(cutout_points,axis=0)[0], torch.min(cutout_points,axis=0)[0])

    upsample = PointcloudUpSampling(max_num_points=3072, centroid="fps")
    new_points = upsample(points)
    print(new_points.shape)
