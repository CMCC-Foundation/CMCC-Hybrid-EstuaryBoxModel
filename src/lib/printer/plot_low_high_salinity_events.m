close all
data = Component4SulPredictions(:,["Sul","Sul_EBM", "Sul_Hybrid_EBM_RF"]);
x = linspace(1,height(data),height(data));

figure()
hold on
patch([0 240 240 0], [0 0 5 5], 'r', 'FaceColor','red','FaceAlpha',.1);
patch([0 240 240 0], [5 5 20 20], 'b', 'FaceColor','blue','FaceAlpha',.1);
patch([0 240 240 0], [20 20 31 31], 'g', 'FaceColor','green','FaceAlpha',.1);

scatter(x, data.Sul);
plot(x, data.Sul_EBM);
plot(x, data.Sul_Hybrid_EBM_RF);

xlim([0 240]);
ylim([0 31]);
xlabel("Record number");
ylabel("Response (psu)");
grid on
hold off

