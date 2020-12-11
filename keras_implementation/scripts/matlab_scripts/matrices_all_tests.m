clear,clc,close all
tic
idpool = [86 36];
for i = 1:2
    id = idpool(i);
    hun = floor(id/100);
    lastt2 = mod(id, 100);
    valid_hr = imread(['Data/BlurryImage_Vol2/output-slice' num2str(hun) num2str(lastt2) '.png']);
    figure(1), imagesc(valid_hr)
    valid_lr = imread(['Data/denoised_dataset_Vol2/output-slice' num2str(hun) num2str(lastt2) '.png']);
    valid_gen = imread(['Data/DnCNN_onebranch_Vol2/output-slice' num2str(hun) num2str(lastt2) '_dncnn.png']);
    valid_bicubic= imread(['Data/DnCNN_synthetic_Vol2/output-slice' num2str(hun) num2str(lastt2) '.png']);
    valid_bm3d = imread(['Data/BM3D_brushlet_Results_Vol2/output-slice' num2str(hun) num2str(lastt2) '.png']);

%     valid_gen = median(valid_gen,3);
%     valid_gen = medfilt2(valid_gen);
%     valid_gen = cat(3, valid_gen, valid_gen,valid_gen);
%     
% %     valid_nogan = median(valid_nogan,3);
% %     valid_nogan = uint8(double(valid_nogan)/max(max(double(valid_nogan)))*255);
% %    valid_nogan = cat(3, valid_nogan, valid_nogan, valid_nogan);
% %     %% PSNR
%     psnrAll(i,1) = psnr(valid_gen,valid_hr); %1:gen, 2:bicubic
%     psnrAll(i,2) = psnr(valid_bicubic,valid_hr);
%     psnrAll(i,3) = psnr(valid_nogan,valid_hr);
%     %% SSIM
%     ssimAll(i,1) = ssim(valid_gen,valid_hr);
%     ssimAll(i,2) = ssim(valid_bicubic,valid_hr);
%     ssimAll(i,3) = ssim(valid_nogan,valid_hr);
%     %% MSE
%     mseAll(i,1) = immse(valid_gen,valid_hr);
%     mseAll(i,2) = immse(valid_bicubic,valid_hr);
%     mseAll(i,3) = immse(valid_nogan,valid_hr);
%     %% correlation
%     corr1 = corrcoef(double(valid_gen),double(valid_hr));
%     correlation(i,1) = corr1(1,2);
%     corr2 = corrcoef(double(valid_bicubic),double(valid_hr));
%     correlation(i,2) = corr2(1,2);
%     corr3 = corrcoef(double(valid_nogan),double(valid_hr));
%     correlation(i,3) = corr3(1,2);

% % %%
% % %i = 4
% % %     x = 186;
% % %     y = 666;
% % %     w_x = 16;
% % %     w_y = 50;
%     x = 186;
%     y = 666;
%     w_x = 16;
%     w_y = 50;
% 
% 
%     valid_lr = repelem(valid_lr, 4, 4);
%     

%     if id == 1
%         x_large = 70;
%         y_large = 94;
%         w_x_large = 30;
%         w_y_large = 20;
% 
%         x_large_2 = 100;
%         y_large_2 = 89;
%         w_x_large_2 = 50;
%         w_y_large_2 = 60;
%     else
%         
%         
%         x_large = 100;
%         y_large = 401;
%         w_x_large = 150;
%         w_y_large = 600;
% 
%         x_large_2 = 100;
%         y_large_2 = 401;
%         w_x_large_2 = 150;
%         w_y_large_2 = 600;
%     end
    
     if i == 1
        x_large = 94;
        y_large = 70;
        w_x_large = 25;
        w_y_large = 20;

        x_large_2 = 100;
        y_large_2 = 120;
        w_x_large_2 = 50;
        w_y_large_2 = 20;
    else
        
        
        x_large = 100;
        y_large = 401;
        w_x_large = 150;
        w_y_large = 600;

        x_large_2 = 100;
        y_large_2 = 401;
        w_x_large_2 = 150;
        w_y_large_2 = 600;
    end
    

    
    figure(1),imagesc(valid_hr),colormap(gray), axis off %,title('hr')
    hold on, rectangle('Position', [y_large x_large w_y_large w_x_large], 'EdgeColor', 'r', 'LineWidth', 3) 
    hold on, rectangle('Position', [y_large_2 x_large_2 w_y_large_2 w_x_large_2], 'EdgeColor', 'y', 'LineWidth', 3) 
    %line([21,121],[140,140],'Color','y','LineWidth',3)  %100 pixel
    print(['noisy_' num2str(id) '.png'],'-dpng')
    
    figure(2),imagesc(valid_lr),colormap(gray), axis off %,title('lr')
    hold on, rectangle('Position', [y_large x_large w_y_large w_x_large], 'EdgeColor', 'r', 'LineWidth', 3) 
    hold on, rectangle('Position', [y_large_2 x_large_2 w_y_large_2 w_x_large_2], 'EdgeColor', 'y', 'LineWidth', 3) 
    print(['hydra_' num2str(id) '.png'],'-dpng')
    
    
    figure(3),imagesc(valid_gen),colormap(gray), axis off%, title('gen')
    hold on, rectangle('Position', [y_large x_large w_y_large w_x_large], 'EdgeColor', 'r', 'LineWidth', 3) 
    hold on, rectangle('Position', [y_large_2 x_large_2 w_y_large_2 w_x_large_2], 'EdgeColor', 'y', 'LineWidth', 3) 
    print(['dncnn_1b_' num2str(id) '.png'],'-dpng')
    
    
    figure(4),imagesc(valid_bicubic),colormap(gray), axis off%, title('bic')
    hold on, rectangle('Position', [y_large x_large w_y_large w_x_large], 'EdgeColor', 'r', 'LineWidth', 3) 
    hold on, rectangle('Position', [y_large_2 x_large_2 w_y_large_2 w_x_large_2], 'EdgeColor', 'y', 'LineWidth', 3) 
    print(['dncnn_syn' num2str(id) '.png'],'-dpng')
    
    
    figure(5),imagesc(valid_bm3d),colormap(gray),axis off% title('bm3d'), 
    hold on, rectangle('Position', [y_large x_large w_y_large w_x_large], 'EdgeColor', 'r', 'LineWidth', 3) 
    hold on, rectangle('Position', [y_large_2 x_large_2 w_y_large_2 w_x_large_2], 'EdgeColor', 'y', 'LineWidth', 3) 
    print(['bm3d_' num2str(id) '.png'],'-dpng')
    
    
    
    %%
    %set 1
%     x = 186;
%     y = 656;
%     w_x = 16;
%     w_y = 60;


    %%
    %
  
    %first inset
    inset_hr = valid_hr(x_large:x_large+w_x_large-1, y_large:y_large+w_y_large-1, 1);
    inset_lr = valid_lr(x_large:x_large+w_x_large-1, y_large:y_large+w_y_large-1, 1);
    inset_gen = valid_gen(x_large:x_large+w_x_large-1, y_large:y_large+w_y_large-1, 1);
    inset_bicubic = valid_bicubic(x_large:x_large+w_x_large-1, y_large:y_large+w_y_large-1, 1);
    inset_bm3d = valid_bm3d(x_large:x_large+w_x_large-1, y_large:y_large+w_y_large-1, 1);
   % figure(6),imagesc(inset_hr),colormap(gray), axis off% ,title('hr')
    %line([2,11],[15,15],'Color','y','LineWidth',3)
    imwrite(inset_hr, ['nosiy_inset' num2str(id) '_1.png'])
    imwrite(inset_lr, ['hydra_inset' num2str(id) '_1.png'])
    imwrite(inset_gen, ['dncnn1b_inset' num2str(id) '_1.png'])
    imwrite(inset_bicubic, ['dncnn_inset' num2str(id) '_1.png'])
    imwrite(inset_bm3d, ['bm3d_inset' num2str(id) '_1.png'])
    
      
    %second inset
    inset_hr = valid_hr(x_large_2:x_large_2+w_x_large_2-1, y_large_2:y_large_2+w_y_large_2-1, 1);
    inset_lr = valid_lr(x_large_2:x_large_2+w_x_large_2-1, y_large_2:y_large_2+w_y_large_2-1, 1);
    inset_gen = valid_gen(x_large_2:x_large_2+w_x_large_2-1, y_large_2:y_large_2+w_y_large_2-1, 1);
    inset_bicubic = valid_bicubic(x_large_2:x_large_2+w_x_large_2-1, y_large_2:y_large_2+w_y_large_2-1, 1);
    inset_bm3d = valid_bm3d(x_large_2:x_large_2+w_x_large_2-1, y_large_2:y_large_2+w_y_large_2-1, 1);
   % figure(6),imagesc(inset_hr),colormap(gray), axis off% ,title('hr')
    %line([2,11],[15,15],'Color','y','LineWidth',3)
    imwrite(inset_hr, ['nosiy_inset' num2str(id) '_2.png'])
    imwrite(inset_lr, ['hydra_inset' num2str(id) '_2.png'])
    imwrite(inset_gen, ['dncnn1b_inset' num2str(id) '_2.png'])
    imwrite(inset_bicubic, ['dncnn_inset' num2str(id) '_2.png'])
    imwrite(inset_bm3d, ['bm3d_inset' num2str(id) '_2.png'])
    
    
    
    
%     figure(7),imagesc(inset_lr),colormap(gray), axis off%,title('lr')
%     print(['_inset' num2str(i) '.png'],'-dpng')
%     figure(8),imagesc(inset_gen),colormap(gray), axis off%, title('gen'),
%     print(['gen_inset' num2str(i) '.png'],'-dpng')
%     figure(9),imagesc(inset_bicubic),colormap(gray), axis off% , title('bic')
%     print(['bic_inset' num2str(i) '.png'],'-dpng')
%     figure(10),imagesc(inset_bm3d),colormap(gray), axis off%, title('bm3d')
%     print(['bm3d_inset' num2str(i) '.png'],'-dpng')
    
    
     %second inset
    
end

mean_psnr = mean(psnrAll)
mean_SSIM = mean(ssimAll)
mean_mse  = mean(mseAll)
mean_corr = mean(correlation)
toc