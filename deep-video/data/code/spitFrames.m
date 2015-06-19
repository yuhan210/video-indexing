listing = dir('F:\\android-app\\data\\videos');
for i = 1: size(listing)
   if ~isempty(strfind(listing(i).name, 'mp4'))
        video_name = strsplit(listing(i).name, '.');
        folder_name = strcat('F:\\android-app\\data\\frames\\', video_name(1));        
        mkdir(folder_name{1});
        mov = VideoReader(strcat('F:\\android-app\\data\\videos\\', listing(i).name));   %# use mmreader on older versions       
        for j = 1: mov.NumberOfFrames
            frame = read(mov,j);     
            %{
            frame = imresize(frame, [480 NaN]);
            [height, width, dim] = size(frame);
            frame = imcrop(frame,[width-640 0 639 480]);     
            %}
            fileName = sprintf('%s\\%d.jpg', folder_name{1}, (j-1));           
            imwrite(frame, fileName ,'Quality',100);
        end
   end
end