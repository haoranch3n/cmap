import numpy as np
from scipy.sparse import csr_matrix
from skimage.segmentation import find_boundaries
from datetime import datetime
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

"""
FUNCTIONS FOR MATCHING CELLS IN 2D SLICES
Author: Haoran Chen and Robert F. Murphy
Version: 1.3 February 14, 2025 R.F.Murphy, Ted Zhang
        Add option for setting JI_max
        Add logging
"""

def get_compartments_diff(arr1, arr2):
	a = set((tuple(i) for i in arr1))
	b = set((tuple(i) for i in arr2))
	diff = np.array(list(a - b))
	return diff


def get_matched_cells(current_cell_arr, new_cell_arr):
	a = set((tuple(i) for i in current_cell_arr))
	b = set((tuple(i) for i in new_cell_arr))
	JI = len(list(a & b)) / len(list(a | b))
	if JI != 0:
		return np.array(list(a)), np.array(list(b)), JI
	else:
		return False, False, False

def append_coord(rlabel_mask, indices, maxvalue):
	masked_imgs_coord = [[[], []] for i in range(maxvalue)]
	for i in range(0, len(rlabel_mask)):
		masked_imgs_coord[rlabel_mask[i]][0].append(indices[0][i])
		masked_imgs_coord[rlabel_mask[i]][1].append(indices[1][i])
	return masked_imgs_coord

def unravel_indices(labeled_mask, maxvalue):
	rlabel_mask = labeled_mask.reshape(-1)
	indices = np.arange(len(rlabel_mask))
	indices = np.unravel_index(indices, (labeled_mask.shape[0], labeled_mask.shape[1]))
	masked_imgs_coord = append_coord(rlabel_mask, indices, maxvalue)
	masked_imgs_coord = list(map(np.asarray, masked_imgs_coord))
	return masked_imgs_coord

def get_coordinates(mask):
	print("Getting cell coordinates...")
	cell_num = np.unique(mask)
	maxvalue = len(cell_num)
	channel_coords = unravel_indices(mask, maxvalue)
	return channel_coords

def compute_M(data):
	cols = np.arange(data.size)
	return csr_matrix((cols, (data.ravel(), cols)),
	                  shape=(data.max() + 1, data.size))

def get_indices_sparse(data):
	M = compute_M(data)
	return [np.unravel_index(row.data, data.shape) for row in M]

def list_remove(c_list, indexes):
	for index in sorted(indexes, reverse=True):
		del c_list[index]
	return c_list

def filter_cells(coords, mask):
	no_cells = []
	for i in range(len(coords)):
		if np.sum(mask[coords[i]]) == 0:
			no_cells.append(i)
	new_coords = list_remove(coords.copy(), no_cells)
	return new_coords

def get_indexed_mask(mask, boundary):
	boundary = boundary * 1
	boundary_loc = np.where(boundary == 1)
	boundary[boundary_loc] = mask[boundary_loc]
	return boundary

def get_boundary(mask):
	mask_boundary = find_boundaries(mask, mode='inner')
	mask_boundary_indexed = get_indexed_mask(mask, mask_boundary)
	return mask_boundary_indexed

def get_mask(cell_list):
	mask = np.zeros((img.shape))
	for cell_num in range(len(cell_list)):
		mask[tuple(cell_list[cell_num].T)] = cell_num
	return mask

def get_new_slice_mask(current_matched_index, new_matched, new_unmatched, max_cell_num, img_current_slice_shape):
	mask = np.zeros((img_current_slice_shape))
	for cell_num in range(len(new_matched)):
		mask[tuple(new_matched[cell_num].T)] = current_matched_index[cell_num]+1
	for cell_num in range(len(new_unmatched)):
		mask[tuple(new_unmatched[cell_num].T)] = max_cell_num + cell_num + 1
	return mask.astype(int)


def get_unmatched_list(matched_list, new_slice_cell_coords):
	total_num = len(new_slice_cell_coords)
	unmatched_list_index = []
	for i in range(total_num):
		if i not in matched_list:
			unmatched_list_index.append(i)
	unmatched_list = []
	for i in range(len(unmatched_list_index)):
		unmatched_list.append(new_slice_cell_coords[unmatched_list_index[i]])
	return unmatched_list, unmatched_list_index

def matching_cells_2D(img, JI_thre):
	"""Match 2D cell labels across consecutive z-slices to build consistent 3D labels.

	Uses a vectorised overlap-matrix approach with optimal (Hungarian) assignment.
	"""
	new_img = [img[0]]
	next_label = int(img[0].max()) + 1

	for slice_num in tqdm(range(1, img.shape[0]), desc="Matching slices"):
		img_current = new_img[slice_num - 1]
		img_new = img[slice_num]

		cur_labels = np.unique(img_current)
		cur_labels = cur_labels[cur_labels > 0]
		new_labels = np.unique(img_new)
		new_labels = new_labels[new_labels > 0]

		n_cur = len(cur_labels)
		n_new = len(new_labels)

		if n_new == 0:
			new_img.append(np.zeros_like(img_new, dtype=np.int32))
			continue

		if n_cur == 0:
			remap_lut = np.zeros(int(new_labels.max()) + 1, dtype=np.int32)
			for lbl in new_labels:
				remap_lut[int(lbl)] = next_label
				next_label += 1
			new_img.append(remap_lut[img_new])
			continue

		# -- vectorised overlap computation -----------------------------------
		cur_flat = img_current.ravel().astype(np.intp)
		new_flat = img_new.ravel().astype(np.intp)

		size_cur_all = np.bincount(cur_flat)
		size_new_all = np.bincount(new_flat)

		max_cur_label = int(cur_labels.max())
		max_new_label = int(new_labels.max())

		cur_lut = np.full(max_cur_label + 1, -1, dtype=np.intp)
		cur_lut[cur_labels] = np.arange(n_cur)
		new_lut = np.full(max_new_label + 1, -1, dtype=np.intp)
		new_lut[new_labels] = np.arange(n_new)

		valid = (cur_flat > 0) & (new_flat > 0)
		overlap_compact = np.zeros((n_cur, n_new), dtype=np.int64)
		if valid.any():
			cur_idx = cur_lut[cur_flat[valid]]
			new_idx = new_lut[new_flat[valid]]
			pair_idx = cur_idx * n_new + new_idx
			overlap_flat = np.bincount(pair_idx, minlength=n_cur * n_new)
			overlap_compact = overlap_flat.reshape(n_cur, n_new)

		# -- Jaccard-index matrix (fully vectorised) --------------------------
		size_cur_vec = size_cur_all[cur_labels]
		size_new_vec = size_new_all[new_labels]
		union_matrix = (size_cur_vec[:, None] + size_new_vec[None, :]
		                - overlap_compact)
		with np.errstate(divide='ignore', invalid='ignore'):
			ji_matrix = np.where(union_matrix > 0,
			                     overlap_compact / union_matrix, 0.0)

		# -- optimal bipartite assignment (Hungarian) -------------------------
		cost = 1.0 - ji_matrix
		row_ind, col_ind = linear_sum_assignment(cost)

		label_map = {}
		matched_new = set()
		for r, c in zip(row_ind, col_ind):
			if ji_matrix[r, c] > JI_thre:
				label_map[int(new_labels[c])] = int(cur_labels[r])
				matched_new.add(int(new_labels[c]))

		# -- remap via LUT ----------------------------------------------------
		remap_lut = np.zeros(max_new_label + 1, dtype=np.int32)
		for new_lbl, cur_lbl in label_map.items():
			remap_lut[new_lbl] = cur_lbl
		for lbl in new_labels:
			lbl_int = int(lbl)
			if lbl_int not in matched_new:
				remap_lut[lbl_int] = next_label
				next_label += 1

		new_img.append(remap_lut[img_new])

	return np.stack(new_img, axis=0)