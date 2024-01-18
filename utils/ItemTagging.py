#!/usr/bin/python3
'''
@Description: Data_handle.py
@Version: 0.0
@Autor: wangding
@Date: 2020-12-10-16:09
@Software:PyCharm
@LastEditors: wangding
@LastEditTime:  2020-12-10-16:09
'''


def BuildItemId(data_name):
    import os

    if os.path.exists(f"./data/{data_name}/cascade_id.txt"): os.remove(f"./data/{data_name}/cascade_id.txt")
    if os.path.exists(f"./data/{data_name}/cascadetest_id.txt"): os.remove(f"./data/{data_name}/cascadetest_id.txt")
    if os.path.exists(f"./data/{data_name}/cascadevalid_id.txt"): os.remove(f"./data/{data_name}/cascadevalid_id.txt")
    total = 300000

    with open(f"./data/{data_name}/cascade.txt", "r") as file_org:
        with open(f"./data/{data_name}/cascade_id.txt", "a") as file_tgt:
            for line in file_org.readlines():
                file_tgt.write(str(total) + '\n')
                total += 1

    with open(f"./data/{data_name}/cascadetest.txt", "r") as file_org:
        with open(f"./data/{data_name}/cascadetest_id.txt", "a") as file_tgt:
            for line in file_org.readlines():
                file_tgt.write(str(total) + '\n')
                total += 1

    with open(f"./data/{data_name}/cascadevalid.txt", "r") as file_org:
        with open(f"./data/{data_name}/cascadevalid_id.txt", "a") as file_tgt:
            for line in file_org.readlines():
                file_tgt.write(str(total) + '\n')
                total += 1


if __name__ == '__main__':

    BuildItemId("twitter")
    BuildItemId("douban")
    BuildItemId("memetracker")