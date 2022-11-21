import csv
import os
import glob
from sklearn.model_selection import train_test_split


def write_csv_from_dict(data_dir, filename, data, ldmk=None):
    with open(data_dir + '/' + filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Patient', 'scan_path'])
        for key, value in data.items():
            writer.writerow([key, value['img']])

# for item in RESULTS:
#     wr.writerow([item,])

def GenDict(data_dir):
    DATA = {}
        
    normpath = os.path.normpath("/".join([data_dir, '**', '*']))

    for img in glob.iglob(normpath, recursive=True):
        if os.path.isfile(img) and True in [ext in img for ext in [".nrrd", ".nii", ".nii.gz", ".mhd", ".dcm", ".DCM", 'gipl.gz']]:
            patient = '_'.join(img.split('/')[-3:-1]).split('_dataset')[0] + '_' + img.split('/')[-1].split('.')[0].split('_scan')[0].split('_Or')[0].split('_OR')[0] #
            if patient not in DATA:
                DATA[patient] = {}
            DATA[patient]['img'] = img

    return DATA

def main(data_dir, output_dir, landmark=None, csv_summary=False):
    # data_dir = args.data_dir

    data = GenDict(data_dir)
    # ic(len(data))

    databis = data
          
    train_ratio = 0.7
    validation_ratio = 0.1
    test_ratio = 0.2

    train, test = train_test_split(list(databis.keys()), test_size=1-train_ratio, random_state=42)   
    val, test = train_test_split(test, test_size=test_ratio/(test_ratio + validation_ratio),random_state=42) 
    
    
    # ic(len(train))
    # ic(len(val))
    # ic(len(test))

    # out_dir = args.out
    
    write_csv_from_dict(output_dir, 'train.csv', {k: databis[k] for k in train})
    write_csv_from_dict(output_dir, 'val.csv', {k: databis[k] for k in val})
    write_csv_from_dict(output_dir, 'test.csv', {k: databis[k] for k in test})

if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description='ALI CBCT Training')
    # parser.add_argument('--data_dir', help='Directory with all data', type=str,default='/home/luciacev/Desktop/Luc_Anchling/DATA/ALI_CBCT/Test')
    # parser.add_argument('--out',help='output directory with csv files',type=str, default='')
    # parser.add_argument('--landmark', help='Landmark that you want to train', type=str,default='S')#required=True)
    # parser.add_argument('--csv_sumup',help='to creat a csv file with scans and the different landmarks that they have',type=bool,default=False)

    # args = parser.parse_args()

    data_dir = '/home/luciacev/Desktop/Luc_Anchling/DATA/ASO_CBCT/Oriented/RESAMPLED'
    output_dir = data_dir + '/CSV'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    main(data_dir, output_dir, landmark=None)