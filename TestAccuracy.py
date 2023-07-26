from glob import iglob 
import os
from utils import LoadOnlyLandmarks
import torch
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff


def search(path,*args):
    """
    Return a dictionary with args element as key and a list of file in path directory finishing by args extension for each key

    Example:
    args = ('json',['.nii.gz','.nrrd'])
    return:
        {
            'json' : ['path/a.json', 'path/b.json','path/c.json'],
            '.nii.gz' : ['path/a.nii.gz', 'path/b.nii.gz']
            '.nrrd.gz' : ['path/c.nrrd']
        }
    """
    arguments=[]
    for arg in args:
        if type(arg) == list:
            arguments.extend(arg)
        else:
            arguments.append(arg)
    return {key: sorted([i for i in iglob(os.path.normpath("/".join([path,'**','*'])),recursive=True) if i.endswith(key)]) for key in arguments}

def GetPatients(test_files,ref_files):
    """To associate scan and json files to every patient in input folder of SEMI ASO"""

    patients = {}

    for i in range(len(test_files)):
        patients = AddInDict(test_files, patients, i, 'test')
        patients = AddInDict(ref_files, patients, i, 'ref')
        
    return patients

def AddInDict(json_files, patients, i,key):
    patientjson = os.path.basename(json_files[i]).split('_Or')[0].split('_OR')[0].split('_lm')[0].split('_Scanreg')[0].split('.')[0]

    if patientjson not in patients.keys():
        patients[patientjson] = {key:json_files[i]}
    else:
        patients[patientjson][key] = json_files[i]
    
    return patients

def MSELoss(args):

    MSE = torch.nn.MSELoss() # Loss

    test_json = search(args.test_dir,'json')['json']
    ref_json = search(args.ref_dir,'json')['json']
    
    patients = GetPatients(test_json,ref_json)

    ldmk_list = LoadOnlyLandmarks(test_json[0]).keys()
    
    TEST = {key:[] for key in ldmk_list}
    REF = {key:[] for key in ldmk_list}

    for patient,data in patients.items():

        if "test" in data.keys() and "ref" in data.keys():

            test = LoadOnlyLandmarks(data["test"],ldmk_list=ldmk_list)
            ref = LoadOnlyLandmarks(data["ref"],ldmk_list=ldmk_list)

            for lm in ldmk_list:
                if lm in test.keys() and lm in ref.keys():
                    TEST[lm].append(test[lm])
                    REF[lm].append(ref[lm])
    
    for lm in ldmk_list:
        print("For landmark: {}".format(lm))
        print(MSE(torch.tensor(np.array(TEST[lm])),torch.tensor(np.array(REF[lm]))))

def normal_vector(data, plane_list):
    points = []
    for lm in plane_list:
        try:
            points.append(data[lm])
        except KeyError:
            continue
    try:    
        X = np.array(points)
        
        U,S,V = np.linalg.svd(X)
        
        return abs(V[-1])
    
    except:
        return np.array([0,0,0])

def gen_plot(NV_test, NV_ref, patient):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.quiver(0, 0, 0, NV_test[0][0], NV_test[0][1], NV_test[0][2], color='r', label='TEST')
    ax.quiver(0, 0, 0, NV_ref[0][0], NV_ref[0][1], NV_ref[0][2], color='b', label='REF')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.legend()
    pitch, yaw, roll = get_pitch_yaw_roll(NV_test[0],NV_ref[0])
    plt.title("{} | Plane 1\nP: {} | Y: {} | R: {}".format(patient,round(pitch,3),round(yaw,3),round(roll,3)))
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.quiver(0, 0, 0, NV_test[1][0], NV_test[1][1], NV_test[1][2], color='r', label='TEST')
    ax.quiver(0, 0, 0, NV_ref[1][0], NV_ref[1][1], NV_ref[1][2], color='b', label='REF')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.legend()
    pitch, yaw, roll = get_pitch_yaw_roll(NV_test[1],NV_ref[1])
    plt.title("{} | Plane 2\nP: {} | Y: {} | R: {}".format(patient,round(pitch,3),round(yaw,3),round(roll,3)))
    plt.show()

def get_pitch_yaw_roll(v1, v2):
    # Step 1: Calculate the cross product of the two vectors
    axis = np.cross(v1, v2)
    
    # Step 2: Calculate the angle between the two vectors
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    sin_theta = np.sqrt(1 - cos_theta**2)
    
    # Step 3: Calculate the rotation matrix
    kx = axis[0]
    ky = axis[1]
    kz = axis[2]
    c = cos_theta
    s = sin_theta
    t = 1 - c
    
    R = np.array([
        [t*kx*kx + c, t*kx*ky - s*kz, t*kx*kz + s*ky],
        [t*kx*ky + s*kz, t*ky*ky + c, t*ky*kz - s*kx],
        [t*kx*kz - s*ky, t*ky*kz + s*kx, t*kz*kz + c]
    ])
    
    # Step 4: Extract pitch, yaw and roll from rotation matrix
    pitch = np.arctan2(R[2][0], np.sqrt(R[0][0]**2 + R[1][0]**2)) * 180 / np.pi
    yaw = np.arctan2(-R[1][0], R[0][0]) * 180 / np.pi
    roll = np.arctan2(-R[2][1], R[2][2]) * 180 / np.pi
    
    return abs(pitch),abs(yaw),abs(roll) #"pitch: {} | roll: {} | yaw: {}".format(abs(pitch),abs(roll),abs(yaw))

def HausdorffDistance(args):
    HAUSDORFF = []
    test_json = search(args.test_dir,'json')['json']
    ref_json = search(args.ref_dir,'json')['json']

    patients = GetPatients(test_json,ref_json)

    df = pd.DataFrame([patient for patient,data in patients.items() if  "test" in data.keys() and "ref" in data.keys()], columns=['Patients'])
    for patient,data in patients.items():
        if "test" in data.keys() and "ref" in data.keys():
            TEST = LoadOnlyLandmarks(data['test'])
            REF = LoadOnlyLandmarks(data['ref'])
            HAUSDORFF.append(max(directed_hausdorff(np.array(list(TEST.values())),np.array(list(REF.values())))[0],directed_hausdorff(np.array(list(REF.values())),np.array(list(TEST.values())))[0]))
    
    df['Hausdorff'] = HAUSDORFF
    
    if not os.path.exists(args.csv_dir):
        os.makedirs(args.csv_dir)
    print(np.mean(HAUSDORFF))
    df.to_csv(os.path.join(args.csv_dir,'Hausdorff.csv'),index=False)


def OrientationError(args,ldmk_list,plane1,plane2):
    CosSim = torch.nn.CosineSimilarity() # /!\ if loss < 0.1 dont apply rotation /!\
    Loss = lambda x,y: abs(CosSim(torch.Tensor(x),torch.Tensor(y)))
    PLANE1_TEST,PLANE2_TEST,PLANE1_REF,PLANE2_REF,NAME,NB_LDMK = [],[],[],[],[],[]

    test_json = search(args.test_dir,'json')['json']
    ref_json = search(args.ref_dir,'json')['json']
    
    patients = GetPatients(test_json,ref_json)

    df = pd.DataFrame([patient for patient,data in patients.items() if  "test" in data.keys() and "ref" in data.keys()], columns=['Patients'])
    for patient,data in patients.items():
        if "test" in data.keys() and "ref" in data.keys() :
            
            test = LoadOnlyLandmarks(data["test"],ldmk_list=ldmk_list)
            ref = LoadOnlyLandmarks(data["ref"],ldmk_list=ldmk_list)
            test1,test2,ref1,ref2 = normal_vector(test,plane1),normal_vector(test,plane2),normal_vector(ref,plane1),normal_vector(ref,plane2)
            
            # print("Patient:",patient)
            pitch1,yaw1,roll1 = get_pitch_yaw_roll(test1,ref1)
            pitch2,yaw2,roll2 = get_pitch_yaw_roll(test2,ref2)
            # print("Plane 1: pitch: {} | roll: {} | yaw: {}".format(round(pitch1,3),round(roll1,3),round(yaw1,3)))
            # print("Plane 2: pitch: {} | roll: {} | yaw: {}".format(round(pitch2,3),round(roll2,3),round(yaw2,3)))
            # print("="*70)
            if pitch1 > 30 or pitch2 > 30 or roll1 > 30 or roll2 > 30 or yaw1 > 30 or yaw2 > 30:
                gen_plot([test1,test2],[ref1,ref2],patient)
            # gen_plot(test2,ref2,patient)
            PLANE1_TEST.append(test1),PLANE2_TEST.append(test2),PLANE1_REF.append(ref1),PLANE2_REF.append(ref2),NAME.append(patient),NB_LDMK.append(len(test))

    # plane1Loss = np.array(Loss(np.array(PLANE1_TEST),np.array(PLANE1_REF)))
    # plane2Loss = np.array(Loss(np.array(PLANE2_TEST),np.array(PLANE2_REF)))
    
    # degplane1 = np.arccos(plane1Loss)*180/np.pi
    # degplane2 = np.arccos(plane2Loss)*180/np.pi
    
    # #df['Plane1'] = plane1Loss
    # df['Degre1'] = degplane1
    # #df['Plane2'] = plane2Loss
    # df['Degre2'] = degplane2
    # df['Nb Landmark'] = NB_LDMK

    # if not os.path.exists(args.csv_dir):
    #     os.makedirs(args.csv_dir)

    # df.to_csv(os.path.join(args.csv_dir,args.csv_name.split('.')[0]+'.csv'),index=False)    
    
    print("="*70)
    print("Plane 1 Accuracy: {}% (Landmarks: {})".format(round(np.mean(plane1Loss)*100,2),plane1))
    print("Plane 2 Accuracy: {}% (Landmarks: {})".format(round(np.mean(plane2Loss)*100,2),plane2))

def GetLandmarks(type_accuracy):
    if type_accuracy == 'head':
        ldmk_list = ['Ba','LOr','ROr','LPo','N','RPo','S']
        plane1 = ['LOr','ROr','LPo','RPo']
        plane2 = ['Ba','N','S']

    if type_accuracy == 'max':
        ldmk_list = ['ANS','PNS','IF','UR6O','UL6O','UR1O','Mid_UR6O_UL6O','Mid_UR1O_UL1O']
        plane1 = ['UR6O','UL6O','UR1O']
        plane2 = ['ANS','PNS','IF','Mid_UR6O_UL6O','Mid_UR1O_UL1O']

    if type_accuracy == 'mand':
        ldmk_list = ['LL6O','LR6O','LR1O','B','Pog','Me']
        plane1 = []
        plane2 = []

    return ldmk_list,plane1,plane2

    # if type_accuracy == 'maxillary':

    # if type_accuracy == 'mandible':

def RootMeanSquareError(args):
    RMSE = []
    test_json = search(args.test_dir,'json')['json']
    ref_json = search(args.ref_dir,'json')['json']
    
    patients = GetPatients(test_json,ref_json)

    df = pd.DataFrame([patient for patient,data in patients.items() if  "test" in data.keys() and "ref" in data.keys()], columns=['Patients'])
    for patient,data in patients.items():
        if "test" in data.keys() and "ref" in data.keys() :
            
            test = LoadOnlyLandmarks(data["test"])
            ref = LoadOnlyLandmarks(data["ref"])
            rmse = np.sqrt(np.mean((np.array(list(test.values()))-np.array(list(ref.values())))**2))
            RMSE.append(rmse)
            print("Patient: {} | RMSE: {}".format(patient,rmse))
    df['RMSE'] = RMSE
    if not os.path.exists(args.csv_dir):
        os.makedirs(args.csv_dir)
    df.to_csv(os.path.join(args.csv_dir,'RMSE.csv'),index=False)
    print("RMSE: {}".format(np.mean(RMSE)))

def main(args):
    print("Test directory: ",args.test_dir)

    ldmk_list,plane1,plane2 = GetLandmarks(args.type_acc)

    RootMeanSquareError(args)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--test_dir', type=str, default='/home/luciacev/Desktop/Luc/DATA/AReg_CBCT/JJ/Approach3/landmarks/T1/MAX/', help='Path to the test directory')
    parser.add_argument('--ref_dir', type=str, default='/home/luciacev/Desktop/Luc/DATA/AReg_CBCT/JJ/Approach3/landmarks/T2/MAX/', help='Path to the reference directory')
    parser.add_argument('--csv_name', type=str, default='Accuracy', help='Name of the csv file')
    parser.add_argument('--csv_dir', type=str, default='./', help='Path to the csv directory')
    parser.add_argument('--type_acc', type=str, help='Type of accuracy to compute', default='max')

    args = parser.parse_args()
    
    main(args)
    