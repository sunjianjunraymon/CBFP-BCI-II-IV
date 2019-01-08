function [out]=pso(in1,in2)

da=max(in1);
xiao=min(in1);

c1=0.014962;
c2=0.014962;
w=0.007298;
MaxDT=7000;
N=80;

for i=1:N
     x(i)=1.5;
     v(i)=rand;    
end

for i=1:N
  p(i)=fitness(in1,in2,x(i));
  y(i)=x(i); 
end

pg=x(1);

for i=2:N
    if fitness(in1,in2,x(i))>fitness(in1,in2,pg)
        pg=x(i);
    end
end

for t=1:MaxDT
    for i=1:N
            v(i)=(w-(w/MaxDT)*t)*v(i)+c1*rand*(y(i)-x(i))+c2*rand*(pg-x(i));
            x(i)=x(i)+v(i);

            for j=1:1
                 if (x(i))>da
                    x(i)=da;
                 end     
                 if (x(i))<xiao
                    x(i)=xiao;
                 end   
            end

            if fitness(in1,in2,x(i))>p(i)
                p(i)=fitness(in1,in2,x(i));
                y(i)=x(i);
            end
            if p(i)>fitness(in1,in2,pg)
                fitness(in1,in2,pg)
                pg=y(i);
            end
    end
    out=pg;
end
   zql=fitness(in1,in2,pg);
end