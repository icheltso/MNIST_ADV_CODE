function img_final = array_to_imgv2(arr)
%Takes row-vectors and returns image arrays. Use with imshow.
    pics = size(arr,1);
    if (pics == 1)
        mnist_type = transpose(reshape(arr,[28,28]));
        img_final = mat2gray(mnist_type);
        return;
    end
    n = ceil(sqrt(pics));
    frst = arr(1,:);
    frst = transpose(reshape(frst,[28,28]));
    for k = 2:n
        temp2 = arr(k,:);
        temp2 = transpose(reshape(temp2,[28,28]));
        frst = cat(2,frst,temp2);
    end
    if (pics == n*n)
        bound = n-1;
    elseif (pics >= (n-1)*n)
        bound = n-2;
    else
        bound = n-3;
    end
    for i = 1:bound
        temp = arr(1+i*n,:);
        temp = transpose(reshape(temp,[28,28]));
        for j = 2:n
            temp2 = arr(j+i*n,:);
            temp2 = transpose(reshape(temp2,[28,28]));
            temp = cat(2,temp,temp2);
        end
        frst = cat(1,frst,temp);
    end
    if (pics == n*n || pics == (n-1)*n)
        %imagesc(frst)
        img_final = mat2gray(frst);
        return;
    end
    if (pics > (n-1)*n)
        rn = pics - n*(n-1);
        dif = n - rn;
        addon = zeros(28,28*dif);
        temp = arr(1+(n-1)*n,:);
        temp = transpose(reshape(temp,[28,28]));
        if (rn == 1)
            temp = cat(2,temp,addon);
            frst = cat(1,frst,temp);
            %imagesc(frst)
            img_final = mat2gray(frst);
            return;
        end
        for j = 2:rn
            temp2 = arr(j+(n-1)*n,:);
            temp2 = transpose(reshape(temp2,[28,28]));
            temp = cat(2,temp,temp2);
        end
        temp = cat(2,temp,addon);
        frst = cat(1,frst,temp);
        %imagesc(frst)
        img_final = mat2gray(frst);
        return;
    end
    rn = pics - n*(n-2);
    dif = n - rn;
    addon = zeros(28,28*dif);
    temp = arr(1+(n-2)*n,:);
    temp = transpose(reshape(temp,[28,28]));
    for j = 2:rn
        temp2 = arr(j+(n-2)*n,:);
        temp2 = transpose(reshape(temp2,[28,28]));
        temp = cat(2,temp,temp2);
    end
    temp = cat(2,temp,addon);
    frst = cat(1,frst,temp);
    %imagesc(frst)
    img_final = mat2gray(frst);
end

