import os
def read_file(path):
    count=0
    for child_dir in os.listdir(path):
        if child_dir == '.DS_Store':
            os.remove(os.path.join(path, child_dir))
        else:
            child_path = os.path.join(path, child_dir)
            right=0
            this = 0
            for child_file in os.listdir(child_path):

                if child_file == '.DS_Store':
                    os.remove(os.path.join(child_path, child_file))
                elif child_file[-3:]!='jpg':
                    os.remove(os.path.join(child_path, child_file))
                else:
                    count+=1
                    this+=1
                if this==20:
                    right=1
            if right==0:
                print(child_path)

    return count

if __name__ == '__main__':
    print(read_file('data/'))