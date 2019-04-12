import os


def readFilename(path, allfile):
    filelist = os.listdir(path)

    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            readFilename(filepath, allfile)
        else:
            allfile.append(filepath)
    return allfile


if __name__ == '__main__':
    path1 = "/home/txl/PycharmProjects/CNN_pytorch/train200/"
    path2 = "/home/txl/PycharmProjects/CNN_pytorch/test200/"

    allfile1 = []
    allfile1 = readFilename(path1, allfile1)
    allname1 = []

    allfile2 = []
    allfile2 = readFilename(path2, allfile2)
    allname2 = []

    txtpath1 = "/home/txl/PycharmProjects/CNN_pytorch/train.txt"
    txtpath2 = "/home/txl/PycharmProjects/CNN_pytorch/test.txt"

    os.remove(txtpath1)
    os.remove(txtpath2)

    for name in allfile1:
        print(name)
        file_cls = name.split("/")[-1].split(".")[-1]
        if file_cls == 'png':
            print(name.split("/")[-1])
            with open(txtpath1, 'a+') as fp:
                F = {'back': 1, 'forward': 2, 'left': 3, 'right': 4}
                fp.write("".join(name) + " " + str(F[name.split("/")[-2]]) + "\n")

    for name in allfile2:
        print(name)
        file_cls = name.split("/")[-1].split(".")[-1]
        if file_cls == 'png':
            print(name.split("/")[-1])
            with open(txtpath2, 'a+') as fp:
                F = {'back': 1, 'forward': 2, 'left': 3, 'right': 4}
                fp.write("".join(name) + " " + str(F[name.split("/")[-2]]) + "\n")




