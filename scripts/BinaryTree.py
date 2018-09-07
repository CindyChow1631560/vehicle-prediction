# -*- coding: utf-8 -*-
"""
Created on Tue May 15 09:30:37 2018

@author: asus
"""

class BinaryTreeNode(object):
    def __init__(self,data=None,left=None,right=None):
        self.data=data
        self.left=left
        self.right=right
        
class BinaryTree(object):
    def __init__(self,root=None):
        self.root=root
        
    def is_empty(self):
        return self.root==None
    
    def preOrder(self,BinaryTreeNode):   #前序遍历
        if BinaryTreeNode==None:
           return 
        print(BinaryTreeNode.data)
        self.preOrder(BinaryTreeNode.left)
        self.preOrder(BinaryTreeNode.right)
       
    def inOrder(self,BinaryTreeNode):   #中序遍历
        if BinaryTreeNode==None:
          return
        self.inOrder(BinaryTreeNode.left)
        print(BinaryTreeNode.data)
        self.inOrder(BinaryTreeNode.right)
        
    def reOrder(self,BinaryTreeNode):
        if BinaryTreeNode==None:
          return
        self.reOrder(BinaryTreeNode.left)
        self.reOrder(BinaryTreeNode.right)
        print(BinaryTreeNode.data)
        
n1 = BinaryTreeNode(data="D")
n2 = BinaryTreeNode(data="E")
n3 = BinaryTreeNode(data="F")
n4 = BinaryTreeNode(data="B", left=n1, right=n2)
n5 = BinaryTreeNode(data="C", left=n3, right=None)
root = BinaryTreeNode(data="A", left=n4, right=n5)

bt=BinaryTree(root)
print("前序遍历")
bt.preOrder(bt.root)
print("中序遍历")
bt.inOrder(bt.root)
print("后序遍历")
bt.reOrder(bt.root)