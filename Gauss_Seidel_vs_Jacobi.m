%Aufgabe_6_5_1
%   Ein konkretes Oligopol-Modell mit N Spielern aus der Quelle
%   Beschreibung:
%     Jedes der N Unternehmen ist ein Spieler \nu
%     jeder Spieler löst das Minimierungsproblem 
%        min_{x^{\nu}}\theta_{\nu} u.d.N x_{\nu}\geq 0
%     Die Auszahlungsfunktion \theta lautet
%        \theta_{\nu}(x) := c_{\nu}(x_{\nu})- ...
%        x_{\nu}p(x_{\nu}+\sum_{\mu\neq\nu}x_{\mu})
%     Hierbei ist c die Kostenfunktion
%        c_{\nu}(x_{\nu}):=c_{\nu}x_{\nu}+\frac{\beta_{\nu}}{1+\beta_{\nu}}
%        L_{\nu}^{-\frac{1}{\beta_{\nu}}}x_{\nu}^
%        {\frac{1+\beta_{\nu}}{\beta_{\nu}}}
%     und p die inverse Nachfragefunktion
%        p(e):=(\frac{\alpha}{e}=^{\frac{1}{\gamma}}
%   
%   
%   Autor: Anne Bayer
%   Datum: 
%   Quelle: C. Kanzow, A. Schwartz, "Spieltheorie", Birkhäuser 2018,
%   S.97-98


%Beispiel aus [Buch] S. []
%% %Konstanten und Werte aus der Quelle
alph=5000;
gamm=1.1;
c_original = [8,6,4,2];%[13,12,10,8,6,4,2];%[18,16,14,12,10,8,6,4,2];
bet_original=[1.1,1.0,0.9,0.8];%[1.2,1.3,1.2,1.1,1.0,0.9,0.8];%[1.6,1.5,1.4,1.3,1.2,1.1,1.0,0.9,0.8];
L_original=ones(1,length(c_original))*5;
eps=1e-4;

N=length(c_original);
x0_original=ones(1,length(c_original))*10; %Startvektor

%%  %Die zu minimierende Funktion
theta = @(x,i,su) c(i).*x + bet(i)./(1+bet(i)).*L(i).^(-1./bet(i)).*x.^((1+bet(i))./bet(i))...
    -x.*(alph./(x+su)).^(1/gamm);


%% Algorithmen 
%Abbruchkriterium: \norm(x_{k-2} -x_{k}) <eps 
%(Als Begründung: Beim Jacobi-Verfahren gibt es bei diesem Beispiel zwei Häufungspunkte)
anz_Iter = zeros(2,N-1);
for met = 1:2
    Algo=met;
    for iteration=2:N
        diff=1+eps; %Abbruchkriterium
        iter=0;
        c=c_original(1:iteration);
        L=L_original(1:iteration);
        bet=bet_original(1:iteration);
        x0=x0_original(1:iteration);
        X=x0;
        X1=x0;
        X2=x0;
        X3=x0+1+eps;
        sumX=x0; %Produktionen von x^{-\nu}(den anderen Spielern)
        for i=1:iteration
            sumX(i) = sum(x0)-x0(i);
        end
        while(diff>eps)
            iter = iter+1;
            if Algo==2
                for i = 1:iteration   
                    summe = sum(x0)-x0(i);          %Immer aktuelle Werte
                    thet = @(x) theta(x,i,summe);   %Aktuelle Auszahlungsfunktion von Spiler i
                    y = fminunc(thet,x0(i));        %Minimalpunkt der Funktion bestimmen
                    x0(i)=y;                        %Aktualisierung des Ausgabvektors       
                end
            end
            if Algo==1         
                for j=1:iteration                           %Einmalige Aktualisierung der Werte pro Iteration
                    sumX(j) = sum(x0)-x0(j);
                end
                for i=1:iteration
                    thet= @(x) theta(x,i,sumX(i));  %Aktuelle Auszahlungsfunktion von Spieler i
                    y=fminunc(thet,x0(i));          %Minimalpunkt der Funktion bestimmen
                    x0(i)= y;                       %Aktualisierung des Ausgabevektors
                end
            end
            %diff=norm(x0-X);                        %Bechnung des Abbruchkriteriums
            diff = norm(X3-X1);
            X=x0;
            X3 = X2;
            X2 = X1;
            X1 = X;
        end
        fprintf('Anzahl Iterationen mit Algorithmus %1.0f: %3.0f  \n  ',Algo, iter);
        disp(X)
        
        anz_Iter(met,iteration)=iter;
    end
end

%% Plotten der Daten und Bild speichern
plot(1:(N),anz_Iter(1,:),1:N,anz_Iter(2,:))

title('Jacobi vs. Gauss-Seidel','fontsize',20)
xlabel('Anzahl Unternehmen','fontsize',20)
ylabel('Anzahl benötigter Iterationen','fontsize',20)
legend('Jacobi','Gauss-Seidel');
saveas(gcf,'JacobiVsGaussSeidel.png')




