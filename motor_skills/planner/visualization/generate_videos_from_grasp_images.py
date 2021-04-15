import os
import itertools

import imageio

EXTENSION = ".jpg"

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def getFirstNonAlphaIndex(string):
    for i, l in enumerate(string):
        if not l.isalpha():
            return i
    return -1

def str2strNflt(string):
    if is_float(string):
        return ('', float(string))
    else:
        mid_ind = getFirstNonAlphaIndex(string)
        if mid_ind == -1:
            print("Incorrectly formatted filename contains string", string)
            sys.exit()
        return (string[:mid_ind], float(string[mid_ind:]))

def main():
    result_path = "/home/mcorsaro/grabstraction_results/"
    img_dir = "door_pca_3"
    vid_path = result_path + '/' + img_dir + '/videos'
    img_path = result_path + '/' + img_dir

    if not os.path.exists(vid_path):
        os.mkdir(vid_path)
    image_filenames = os.listdir(img_path)
    image_filenames.remove('videos')

    def parseFilename(filename):
        f_no_ex = filename.replace(EXTENSION, '')
        pos, labels = [], []
        sf = f_no_ex.split('_')
        var = int(sf[0])
        dim_vals_with_labels = [str2strNflt(val) for val in sf[1:]]
        return (filename, var, dim_vals_with_labels)

    parsed_filenames_by_var = {}
    for fn in image_filenames:
        parsed_filename = parseFilename(fn)
        if parsed_filename[1] not in parsed_filenames_by_var:
            parsed_filenames_by_var[parsed_filename[1]] = [parsed_filename]
        else:
            parsed_filenames_by_var[parsed_filename[1]].append(parsed_filename)
    dimensions_to_vary = list(parsed_filenames_by_var.keys())
    dimensions_to_vary.sort()

    print("Found", [(key, len(parsed_filenames_by_var[key])) for key in dimensions_to_vary], "images in", img_path)
    for dim in dimensions_to_vary:
        video = imageio.get_writer(vid_path + '/v' + str(dim) + '.mp4', format='FFMPEG', mode='I', fps=15,
                       codec='h264_vaapi',
                       output_params=['-vaapi_device',
                                      '/dev/dri/renderD128',
                                      '-vf',
                                      'format=gray|nv12,hwupload'],
                       pixelformat='vaapi_vld')

        parsed_filenames = parsed_filenames_by_var[dim]
        unique_val_sets = []
        for i in range(len(dimensions_to_vary)):
            unique_val_sets.append(set())
        for filepath in parsed_filenames:
            for i, dimval in enumerate(filepath[2]):
                unique_val_sets[i].add(dimval)
        unique_val_lists = [list(tset) for tset in unique_val_sets]
        print("When varying", dim, "each dim contains", [len(l) for l in unique_val_lists])

        for l in unique_val_lists:
            l.sort(key = lambda x: (x[1], x[0]))

        dim_vals_to_vary = unique_val_lists[dim]
        # list of lists with dim vals len(num_dim-1)
        other_dim_vals = unique_val_lists[:dim] + unique_val_lists[dim+1:]
        combinations_of_other_dim_vals = list(itertools.product(*other_dim_vals))
        for other_dim_val_combination in combinations_of_other_dim_vals:
            for varied_val in dim_vals_to_vary:
                image_filelist = [None]*len(dimensions_to_vary)
                for i in range(len(dimensions_to_vary)):
                    if i < dim:
                        image_filelist[i] = other_dim_val_combination[i]
                    elif i == dim:
                        image_filelist[i] = varied_val
                    elif i > dim:
                        image_filelist[i] = other_dim_val_combination[i-1]
                image_filename = str(dim) + '_'
                for item in image_filelist:
                    image_filename += item[0] + "{:.9f}".format(item[1]) + '_'
                image_filename = image_filename[:-1]
                image_filename += EXTENSION

                latest_image = imageio.imread(img_path + '/' + image_filename)
                video.append_data(latest_image)
                # draw image_filelist
            # wait an extra second
            video.append_data(latest_image)
            video.append_data(latest_image)
        video.close()

if __name__ == '__main__':
    main()