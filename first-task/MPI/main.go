package main

import (
	"fmt"
	mpi "github.com/marcusthierfelder/mpi"
)

func main() {
	mpi.Init()
	defer mpi.Finalize()

	rankCurrentProcess := mpi.Comm_rank(mpi.COMM_WORLD)
	numberProcesses := mpi.Comm_size(mpi.COMM_WORLD)

	fmt.Printf("Процесс %v из %v готов к работе\n", rankCurrentProcess, numberProcesses)

	if rankCurrentProcess != 0 {
		mess := fmt.Sprintf("Процесс %v приветствует главный!", rankCurrentProcess)
		data := []rune(mess)

		length := len(data)
		mpi.Send_int([]int{length}, 0, 0, mpi.COMM_WORLD)

		mpi.Send(data, 0, 1, mpi.COMM_WORLD)
	} else {
		for otherRankProcess := 1; otherRankProcess < numberProcesses; otherRankProcess++ {
			length := make([]int, 1)
			mpi.Recv_int(length, otherRankProcess, 0, mpi.COMM_WORLD)

			data := make([]rune, length[0])

			mpi.Recv(data, otherRankProcess, 1, mpi.COMM_WORLD)
			fmt.Printf("Получено от %v процесса: %s\n", otherRankProcess, string(data))
		}

		fmt.Printf("Всего обработано сообщений: %v\n", numberProcesses-1)
	}
}

/*package main

import (
	"fmt"
	"github.com/btracey/mpi"
)

func main() {

	mpi.Init()
	defer mpi.Finalize()

	rankCurrentProcess := mpi.Rank()
	numberGlobalProcess := mpi.Size()
	fmt.Println(rankCurrentProcess)

	if rankCurrentProcess != 0 {
		message := fmt.Sprintf("Процесс %v приветствует главный!", rankCurrentProcess)
		err := mpi.Send(message, 0, 0)
		if err != nil {
			panic(err)
		}
	} else {
		for rankOtherProcess := 1; rankOtherProcess < numberGlobalProcess; rankOtherProcess++ {
			var received string
			for err := mpi.Receive(&received, rankOtherProcess, 0); err == nil; err = mpi.Receive(&received, 0, 0) {
			}
			fmt.Println(received)
		}
	}
}
*/
