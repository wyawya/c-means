clc
clear
close all
img = double(imread('C:\Users\zhangbohang\Desktop\temp.jpg'));
subplot(1,2,1),imshow(img,[]);
data = img(:);
%分成4类
[center,U,obj_fcn] = fcm(data,2);
[~,label] = max(U); %找到所属的类
%变化到图像的大小
img_new = reshape(label,size(img));
subplot(1,2,2),imshow(img_new,[]);