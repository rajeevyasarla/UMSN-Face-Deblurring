close all;
clear all;
clc;
%Download Test Data provided by authors Ziyi et al. from,
%https://sites.google.com/site/ziyishenmi/cvpr18_face_deblur
listinfo = dir('./Test_data_Helen/final_Helen_gt');
m = length(listinfo);
count =1;
for i = 3:m
    imgname = strcat('./Test_data_Helen/final_Helen_gt','/',listinfo(i).name);
    tmp = listinfo(i).name;
    tmp1 = split(tmp,'.');
    I = imread(imgname);
    im_name = tmp1{1};
    for j = 1:10
        for k = 13:2:27
            blrname = strcat('./Test_data_Helen/final_Helen_blur/',im_name,'_ker',num2str(j,'%02d'),'_blur_k',num2str(k),'.png');
            B = imread(blrname);
            blrname = strcat('./final_Helen_result/',im_name,'_ker',num2str(j,'%02d'),'_blur_k',num2str(k),'_random.png');
            S = imread(blrname);
            filename = strcat('./Testh_gt/',num2str((count),'%06d'),'.png');
            imwrite(I,filename);
            filename = strcat('./Testh/',num2str((count),'%06d'),'.png');
            imwrite(B,filename);
            filename = strcat('./Testh_st/',num2str((count),'%06d'),'.png');
            imwrite(S,filename);
            count = count+1;
        end
    end
    i
end
