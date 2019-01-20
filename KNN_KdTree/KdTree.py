R = 2# Feature dimension

class Node(object):
    def __init__(self, elem, lchild=None, rchild=None):
        self.elem = elem
        self.lchild = lchild
        self.rchild = rchild

def KDT(root, LR, dataSource, r):
    if dataSource==[]:
        return
    data = sorted(dataSource, key=lambda x:x[r])
    r = (r+1) % R
    length = len(data)
    node = Node(data[length//2], None, None)
    if LR==0:
        root.lchild = node
        KDT(root.lchild, 0, data[:length//2], r)
        KDT(root.lchild, 1, data[length//2 + 1:], r)
    if LR==1:
        root.rchild = node
        KDT(root.rchild, 0, data[:length//2], r)
        KDT(root.rchild, 1, data[length//2 + 1:], r)


def InitTree(dataSource, length):
    r = 0
    if dataSource==[]:
        print("Please init dataSource.")
        return None

    data = sorted(dataSource, key=lambda x:x[r])   #使用x[0]，也就是每个元素的x值进行排序
    r=(r+1) % R  #(0+1)%3=1,也就是用y值进行排序切分
    root = Node(data[length//2], None, None)  #创建根节点

    KDT(root, 0, data[:length//2], r)  #0表示左子树
    KDT(root, 1, data[length//2 + 1:], r) #1表示右子树

    print("InitTree Done.")
    return root

def PreOrderTraversalTree(root):
    if root:
        print(root.elem,' | ',)
        PreOrderTraversalTree(root.lchild)
        PreOrderTraversalTree(root.rchild)

def InOrderTraversalTree(root):
    if root:
        InOrderTraversalTree(root.lchild)
        print(root.elem,' | ',)
        InOrderTraversalTree(root.rchild)

def PostOrderTraversalTree(root):
    if root:
        PostOrderTraversalTree(root.lchild)
        PostOrderTraversalTree(root.rchild)
        print(root.elem,' | ',)


if __name__ == "__main__":
    dataSource = [(2,3, 100), (5,4,70), (9,6,55), (4,7,200), (8,1,44), (7,2,0)]
    length = len(dataSource)
    root = InitTree(dataSource, length)

    print("PreOrder:")
    PreOrderTraversalTree(root)
    print("\nInOrder:")
    InOrderTraversalTree(root)
    print("\nPostOrder:")
    PostOrderTraversalTree(root)


# 输出结果：
# InitTree Done.
# PreOrder:
# (7, 2, 0)  |
# (5, 4, 70)  |
# (2, 3, 100)  |
# (4, 7, 200)  |
# (9, 6, 55)  |
# (8, 1, 44)  |
#
# InOrder:
# (2, 3, 100)  |
# (5, 4, 70)  |
# (4, 7, 200)  |
# (7, 2, 0)  |
# (8, 1, 44)  |
# (9, 6, 55)  |
#
# PostOrder:
# (2, 3, 100)  |
# (4, 7, 200)  |
# (5, 4, 70)  |
# (8, 1, 44)  |
# (9, 6, 55)  |
# (7, 2, 0)  |