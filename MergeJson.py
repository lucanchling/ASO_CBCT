import json 
import os
import glob
import argparse

def main(args):
    
    data_dir = args.data_dir

    # if os.path.exists(out_dir):
    #     os.system('rm -rf ' + out_dir)
    # os.mkdir(out_dir)
    
    normpath = os.path.normpath("/".join([data_dir, '**', '']))
    json_file = [i for i in sorted(glob.iglob(normpath, recursive=True)) if i.endswith('.json')]

    # ==================== ALL JSON classified by patient  ====================
    dict_list = {}
    for file in json_file:
        patient = '_'.join(file.split('/')[-3:-1])+'#'+file.split('/')[-1].split('.')[0].split('_lm')[0]+'_lm'
        if patient not in dict_list:
            dict_list[patient] = []
        dict_list[patient].append(file)

    # ==================== MERGE JSON  ====================``
    for key, files in dict_list.items():
        file1 = files[0]
        with open(file1, 'r') as f:
            data1 = json.load(f)
            data1["@schema"] = "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json#"
        for i in range(1,len(files)):
            with open(files[i], 'r') as f:
                data = json.load(f)
            data1['markups'][0]['controlPoints'].extend(data['markups'][0]['controlPoints'])
        outpath = os.path.normpath("/".join(files[0].split('/')[:-1]))        # Write the merged json file
        with open(outpath+'/'+key.split('#')[1] + '_'+ args.extension +'.mrk.json', 'w') as f: #out_dir + '/' + key.split('#')[0].split('_dataset')[0] + '_' + key.split('#')[1] + '_MERGED.mrk.json', 'w') as f:
            json.dump(data1, f, indent=4)

    # ==================== DELETE UNUSED JSON  ====================
    for key, files in dict_list.items():
        for file in files:
            if args.extension not in os.path.basename(file):
                os.remove(file)    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',help='directory where json files to merge are',type=str,default='/home/luciacev/Desktop/Luc_Anchling/DATA/ASO_CBCT/TESTMERGED1')
    parser.add_argument('--extension',help='extension of new merged json files',type=str,default='MERGED')
    args = parser.parse_args()

    main(args)