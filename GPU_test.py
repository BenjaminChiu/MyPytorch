import torch

print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.__version__)
#
# print(1e-1)


# class Mycontex(object):
#     def __init__(self,name):
#         self.name=name
#         print("进入init函数")
#     def __enter__(self):
#         print("进入enter")
#         return self
#     def do_self(self):
#         print(self.name)
#     def __exit__(self,exc_type,exc_value,traceback):
#         print("退出exit")
#         print(exc_type,exc_value)
#
# if __name__ == '__main__':
#     with Mycontex('test') as mc:
#         mc.do_self()


tensor = torch.rand(5, 3)

x = torch.ones(2, 2, requires_grad=True)

y = x + 2
y.grad_fn

z = y * y * 3
out = z.mean()

out.backward()

print(str(tensor))
print('x:'+str(x))
print('y:'+str(y))
print('z:'+str(z))
print('out:'+str(out))
print('x.grad:'+str(x.grad))
