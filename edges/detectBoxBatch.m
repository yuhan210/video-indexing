%% Detect bbx in batch %%

%% load pre-trained edge detection model and set opts (see edgesDemo.m)
model=load('models/forest/modelBsds'); model=model.model;
model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;

%% set up opts for edgeBoxes (see edgeBoxes.m)
opts = edgeBoxes;
opts.alpha = .65;     % step size of sliding window search
opts.beta  = .75;     % nms threshold for object proposals
opts.minScore = .01;  % min score of boxes to detect
opts.maxBoxes = 500;  % max number of boxes to detect

fh = fopen('/mnt/video_list.txt');
videos = textscan(fh, '%s', 'Delimiter', '\n');
videos = videos{1};

fclose(fh);

parfor vid = 1: length(videos)
    
    video_name = videos{vid};
    disp(video_name)
    frame_folder = strcat('/mnt/frames/', video_name);
    
    output_folder = strcat('/mnt/tags/edgebox-all/', video_name, '/');
    if (exist(output_folder, 'dir') == 0), mkdir(output_folder); end
    
    dir_data = dir(frame_folder);
    
    for i = 3:length(dir_data)
        frame_name = dir_data(i).name;
        frame_path = strcat(frame_folder, '/', frame_name);
        %disp(frame_path)
        
        % detect bbx
        I = imread(frame_path);
        tic, bbs=edgeBoxes(I,model,opts); toc
        disp(toc)
                
        
        %% write bbx into a file        
        outfile_path = strcat('/mnt/tags/edgebox-all/', video_name, '/' , frame_name(1:length(frame_name)-4), '.bbx');
        fid=fopen(outfile_path,'w'); assert(fid>0);        
        
        [th, tw] = size(bbs);
        fprintf(fid, 'Exec Time(sec): %f\n', toc);
        for h = 1:th            
            fprintf(fid, '%d %d %d %d %f\n', bbs(h,1), bbs(h,2), bbs(h,3), bbs(h,4), bbs(h,5));
          
        end
        fclose(fid);
        
        
    end
   
    
end

