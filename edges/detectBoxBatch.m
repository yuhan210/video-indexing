% Detect bbx in batch

%% load pre-trained edge detection model and set opts (see edgesDemo.m)
model=load('models/forest/modelBsds'); model=model.model;
model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;

%% set up opts for edgeBoxes (see edgeBoxes.m)
opts = edgeBoxes;
opts.alpha = .65;     % step size of sliding window search
opts.beta  = .75;     % nms threshold for object proposals
opts.minScore = .01;  % min score of boxes to detect
opts.maxBoxes = 1e4;  % max number of boxes to detect

%% detect Edge Box bounding box proposals (see edgeBoxes.m)


I = imread('/mnt/frames/baby_turtle_eating_a_raspberry_vb11rZgfihc/5.jpg');
tic, bbs=edgeBoxes(I,model,opts); toc

%% write bbx into a file
fName = 
fid=fopen(fName,'w'); assert(fid>0);
fprintf(fid,'%% bbGt version=%i\n',vers);
objs=set(objs,'bb',round(get(objs,'bb')));
objs=set(objs,'bbv',round(get(objs,'bbv')));
objs=set(objs,'ang',round(get(objs,'ang')));
for i=1:length(objs)
  o=objs(i); bb=o.bb; bbv=o.bbv;
  fprintf(fid,['%s' repmat(' %i',1,11) '\n'],o.lbl,...
    bb,o.occ,bbv,o.ign,o.ang);
end
fclose(fid);