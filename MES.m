function [score expand]=MES(seed,deta,seedtemp,train)
    a=seed;
    a(:,a(1,:)==0)=[];  %��һ����0ȫ������
    a=a';
    expand=repmat(a,1,size(train,2));%�Ѿ����������е���
          
    for i=1:size(expand,2)
        he(i)=columnentropy(expand(:,i),i,seedtemp);
    end
    he=he(he>0);
    score=mean(he);
   
    delzero=expand;
    
    while ( score>deta  &&  size(delzero,2)>2  )    
        
        for i=1:size(expand,1)
                temp=expand;
                temp(i,:)=0;
                %���¶�tempÿ����entropy,�ٽ�ÿ�м�������
               for j=1:size(temp,2)
               %  he1(j)=columnentropy(expand(:,j),j,seedtemp);
                  he2(j)=columnentropy(temp(:,j),j,seedtemp);
               end
               he2=he2(he2>0);
               scorer=mean(he2);
               er(i)=scorer;%û��ɾ����һ����ǰ��entropy��ȥɾ����һ���Ժ��entropy        
        end
        tempr=score-er;
        [vr, posr]=max(tempr);
              
         for i=1:size(expand,2)
             temp=expand;
             temp(:,i)=0;
              for j=1:size(temp,2)
                 hec(j)=columnentropy(temp(:,j),j,seedtemp); 
              end
               hec=hec(hec>0);
               scorec=mean(hec);
               ec(i)=scorec;
         end
         tempc=score-ec;
         [vc, posc]=max(tempc);
                 
         
         %�Ƚ��м��д�С��ɾ���ϴ�ġ�          
         if vr>vc %delete row
            expand(posr,:)=0;
         end
         
         if vr==vc %delete row
            expand(posr,:)=0;
         end
         
         if vr<vc %delete column
            expand(:,posc)=0;
         end
         
          %�����������ɾ��һ�л���һ���Ժ�Ĳв�÷�
           for i=1:size(expand,2)
              he3(i)=columnentropy(expand(:,i),i,seedtemp);
           end
           he3=he3(he3>0); 
           score=mean(he3);
           %��ȫ����ɾ��
            delzero=expand;
            delzero(:,find(sum(abs(delzero),1)==0))=[];
         
    end
    
      expand (all(expand == 0, 2),:) = [];
      
      if  size(expand,1)<5
            expand=[];
      end
      
      expcopy=expand;
      expcopy (:,all(expcopy == 0, 1)) = [];
%       if size(expcopy,2)>5
%          expand=[]; 
%       end
      
end




%ת��˼·��expand��ÿ��Ԫ�ض������������������������31�У�����Ԫ�ر�Ϊ0