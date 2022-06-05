clear all;
close all;

% loop through all files 
files = split(ls('/home/leo/Skrivbord/normalized_images/'));
disp(files)
for f=1:length(files)
    
    tic
    
    clear inst_centroid;
    clear inst_map;
    file = files(f,:);
    path = "/home/leo/Skrivbord/normalized_images/" + file;
    file_name = erase(file, '.png');
    
    %% read in data


    im_raw = imread(path);
    disp(path)
    rgbImage = cat(3, im_raw, im_raw, im_raw);
    imwrite(rgbImage,string(append(file_name, '.png')));

end
%     figure();
%     imshow(im_raw);
%{

    % annotated image (just has the different instances as blobs)
    im = imread(append("/home/leo/Skrivbord/annotations/", file));

%     figure();
%     imshow(im);

    %% step 1: convert image to binary image
    im_gray = rgb2gray(im);

%     figure();
%     imshow(im_gray);

    %% step 2: create instance map based on binary image 
    % --> create matrix called "inst_map"
     [x,y] = size(im_gray);  
     inst_map = zeros(x,y);   % create empty instance map

     label_dict = [0 0 0 0]; % label dictionary (old labels (r g b) - new labels)

     counter = 1; % counts number of segments and uses this as label

     % fill instance map by iterating through image
     for i=1:x
         for j=1:y

             old_pixel = squeeze(im(i,j,:))'; % get old pixel value. Unfortunately the blob might not have unique colors

             members = ismember(label_dict(:,1:3),old_pixel,'rows');

             if nnz(members)  % check if we already assigned a label to this blob

                 idx = find(members==1);  % get the label

                 new_label = label_dict(idx,4);  % unique color!
                 inst_map(i,j) = new_label;  % put the correct label into the instance map

             else
                 new_label = counter;
                 label_dict = [label_dict; [old_pixel(1) old_pixel(2) old_pixel(3) new_label]]; % append to label dictionary
                 counter = counter + 1;

                 inst_map(i,j) = new_label;
             end  

         end
     end

%     figure();
%     imshow(label2rgb(inst_map));


    %% step 3: calculate the centroids

    % 3.1: We iterate through blobs and mask the image so just the blob with
    % label x i still visible 
    
    inst_centroid = zeros(length(counter)-1,2);
    for k=1:counter-1  % counter has the number of labels we have

        mask = inst_map==k;
        %figure();
        %imshow(mask);

        % 3.2: We calculate the centroid of that blob with regionprops
        centroid = regionprops(mask,'centroid');
        inst_centroid(k,:) = centroid(1).Centroid;


        % 3.3: We extract the centroids coordinates and store them in the matrix.
        % Now they are ordered based on labels

    end


    figure();
    imshow(label2rgb(inst_map));
    hold on 
    plot(inst_centroid(:,1)',inst_centroid(:,2)','o','MarkerEdgeColor',[0 0 0],...
                  'MarkerFaceColor',[1 1 1],...
                  'LineWidth',1.5)
    hold off



    % we cannot do the inst_type and type_map, as we do not have any
    % information about the cell types


    %% store result in .mat files
    % should contain the inst_map and inst_centroid
    save(string(append(file_name, '.mat')),'inst_map','inst_centroid')
    toc
end
%}
