import os
import glob
import ntpath
foto_zip_dir = os.path.normpath('C:\\Users\\demeter\\Desktop\\praca\\amr\\veolia-amr\\data\\annotated\\foto_zip\\')
foto_zip_files = set(glob.iglob(os.path.join(foto_zip_dir, '*.jpg')))
exclude_files = set(glob.iglob(os.path.join(
    'C:\\Users\\demeter\\Desktop\\praca\\amr\\veolia-amr\\data\\annotated\\all_annotated', '*.jpg')))
dest_dir = '../images/test'

exclude_files = {ntpath.basename(path) for path in exclude_files}
print('# jpgs', len(foto_zip_files))
for jpg_path in foto_zip_files:
    if ntpath.basename(jpg_path) not in exclude_files:
        pass
    else:
        print('already there')
