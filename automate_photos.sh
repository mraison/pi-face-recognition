for i in 0 1 2 3 4 5 6 7 8 9
do
	raspistill -o ./dataset/matthew_raison/photo_0${i}.jpg &
	BACK_PID=$!
	wait $BACK_PID
done
