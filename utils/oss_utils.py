'''
Author: wangyue
Date: 2023-02-06 18:06:08
LastEditTime: 2023-02-06 18:06:08
Description:
训练模型相关工具类
'''
import oss2

class OssUtils:
    def __init__(self,access_key_id='LTAI5tNngauABWNaAZUbUALS',access_key_secret='HEKrKUBzaZKCN0An0TdJIRxCS9mdoz',
                endpoint='https://oss-cn-beijing-internal.aliyuncs.com',bucket_name='haomo-generalization'):
        self.oss_bucket = oss2.Bucket(oss2.Auth(access_key_id, access_key_secret),endpoint, bucket_name)
    
    def upload_file(self,oss_full_file_name,local_full_file_name):
        try:
            # self.oss_bucket.put_object('m1/dataset/csv/0131','')
            put_status = self.oss_bucket.put_object_from_file(oss_full_file_name, local_full_file_name)
            if 200 == put_status.status:
                print(f"---------------------上传云oss【原始文件是{local_full_file_name}】成功---------------------")
            else :
                print(f"---------------------上传云oss【原始文件是{local_full_file_name}】失败---------------------")
        except Exception as e:
            print('upload file fail',e)
            
    
    def download_file(self,oss_full_file_name,local_full_file_name):
        try:
            get_result = self.oss_bucket.get_object_to_file(oss_full_file_name,local_full_file_name)
            if 200 == get_result.status:
                print(f"---------------------下载云oss【原始文件是{oss_full_file_name}】成功---------------------")
            else :
                print(f"---------------------下载云oss【原始文件是{oss_full_file_name}】失败---------------------")
        except Exception as e:
            print('download file fail',e)

if __name__=='__main__':
    oss_full_file_name = 'm1/dataset/csv/0131/pretrain_seq2seq_03.csv'
    # oss_full_file_name = 'pretrain_seq2seq_02.csv'
    local_full_file_name = '/data3/data_haomo/m1/csv/0131/pretrain_seq2seq_03.csv'
    oss_utils = OssUtils()
    
    # oss_utils.upload_file(oss_full_file_name,local_full_file_name)
    oss_utils.download_file(oss_full_file_name,local_full_file_name)
    
