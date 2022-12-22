train = load("train.csv");

unsurvived = train(find(train(:,3)== 0),:);
survived = train(find(train(:,3)== 1),:);

figure
plot(log(1+survived(:,1)),log(1+survived(:,2)),"*");
hold on
plot(log(1+unsurvived(:,1)),log(1+unsurvived(:,2)),"+");
hold off
xlabel("Log Age")
ylabel("Log Ticket Fare")
legend("Not Survived", "Survived")
title("Survived Passenger By Age And Ticket Fare")

