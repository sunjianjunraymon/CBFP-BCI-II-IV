function he = columnentropy( column ,k,seedtemp)

    %���һ��ȫΪ0����ô����
    he=0;
    if length(find(column==0))~=size(column,1)
                   %columnһ��Ҫ�������� 
                    posclu=zeros(1,size(column,1));
                     temp=seedtemp(k).real;
                    for i=1:size(column,1)             
                        num=0;
                        for m=1:size(temp,1) %����
                            for n=1:size(temp,2) %����
                                %�����д�����Ϊ���۲�ξ����ʱ����Щɾ���ˣ�ɾ����ÿ���Գ�һ��
                               if temp(m,n)== column(i,1)
                                   posclu(1,i)=m;
                                   num=1;
                               end
                             end
                        end

                        if num==0
                            posclu(1,i)=(i+2000);
                        end

                    end
                    x=posclu';

                    %xһ��Ҫ�������� 
                    x=sort(x);
                    d=diff([x;max(x)+1]);
                    count = diff(find([1;d])) ;
                    y =[x(find(d)) count];

                    lengthzero=0;

                    %���ɾ��һ�У���Ӧ����121����Ϊ���ó�0��Ϊ0�Ĳ��������
                    if y(1,1)==0
                        lengthzero=y(1,2);
                        y=y(2:end,:);
                    end

                    tempentropy=(y(:,2)/(size(column,1)-lengthzero));

                    he=0;
                    for i=1:size(y,1)                     
                        tempc=tempentropy(i,1);
                        he=he-tempc*log2(tempc);
                    end

    end   
    if he<0
        he
    end
  
end

