import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;

/**
 * Tournament.java - Simple SaintPetersburg tournament software
 * See the TODO comments below.  This software assumes that all player class files are located in the same directory and have a unique player ID as a prefix
 * to their SPPlayer subclasses.  List these unique player classnames in the String[] playerIdentifiers.
 * The software will create detailed player logs of games in a "log/" subdirectory, naming files according to the 0-based index of 
 * player numbers in playerIdentifiers.
 * A SPResults.csv file will be generated summarizing tournament results.
 * While running, the console will report high-level progress in the round-robin tournament.
 * If a round-robin tournament is not desired (e.g. too many players), please develop alternative tournament code and share it!
 * @author Todd W. Neller
 * 
 * On our system: java -cp "/usr/local/smile/lib/*:." SPTournament > tournament.log &
 * 
 */
public class SPTournament {

	private static final int NUM_GAMES = 99; // TODO - set the number of games per match
	// private static final long MILLISECONDS_PER_GAME = 30000; // TODO - set the maximum milliseconds per game
	// (300000 ms/game = 300 seconds/game = 5 minutes/game)
    // Note: We will not enforce this time limit in the first tournament, but may later as there is greater freedom 
    // for player search modifications.

	/**
	 * Run a round-robin SaintPetersburg tournament with given hard-coded parameters noted with "TODO" comments.
	 * @param args (unused)
	 * @throws Exception An exception is raised if there is a difficulty writing out results to files.
	 */
	public static void main(String[] args) throws Exception {
		long startTime = System.currentTimeMillis();

		// TODO - List class names for the competing SPPlayer class file names.
//		String[] playerIdentifiers = {"SPPlayerFlatMC", "AIDanSPPlayerHybridObjective", "JFMTPlayer", "MaiNePlayer", "MiMaPlayer", "OKHybridPlayer"};
		String[] playerIdentifiers = {"SPPlayerFlatMC", "AIDanSPPlayerHybridObjective", "MaiNePlayer"};

		String[] competitors = new String[playerIdentifiers.length];
		for (int i = 0; i < playerIdentifiers.length; i++)
			competitors[i] = playerIdentifiers[i] ;
		File logFolder = new File("logs");
		logFolder.mkdir();
		File results = new File("SPTournamentResults.csv");
		PrintWriter resultsOut = new PrintWriter(results); // output to CSV results file
		int[] wins = new int[competitors.length];
		int[] totalWins = new int[wins.length];
		int[] points = new int[wins.length];

		resultsOut.println("\"TOURNAMENT RESULTS:\"");
		resultsOut.println("\"First Player (Win:Loss) Ratios:\"");
		resultsOut.println(",\"Second Player\"");
		resultsOut.print("\"First Player\"");
		for (int i = 0; i < competitors.length; i++) {
			resultsOut.print(",\"" + i + "-" + playerIdentifiers[i] + "\"");
		}
		resultsOut.println();
		for (int i = 0; i < competitors.length; i++) {
			resultsOut.print("\"" + i + "-" + playerIdentifiers[i] + "\"");
			for (int j = 0; j < competitors.length; j++) {
				resultsOut.print(",");
				if (i >= j) { // Don't play a player against itself or against an opponent already played.
                              // Also, player order does not matter in Saint Petersburg, so skip half the matches.
					continue;
				}
				//individual matches recorded in detail in logs directory for later inspection
				String logFilename =  "logs/tournament." + i + "." + j + ".log";    
				PrintWriter logOutput = new PrintWriter(new File(logFilename)); // Create log file for play details.
				System.out.println("Playing games with players 1:" + playerIdentifiers[i] + ", 2:" + playerIdentifiers[j] + ":"); // Console high-level progress
				int[] scores = runGames(competitors[i], competitors[j], logOutput);
				logOutput.flush();
				logOutput.close();
				if (scores[0] > scores[1])
					wins[i]++;
				if (scores[1] > scores[0])
					wins[j]++;
				totalWins[i] += scores[0];
				totalWins[j] += scores[1];
				points[i] += scores[2];
				points[j] += scores[3];

				resultsOut.print("\"(" + scores[0] + ":" + scores[1] + ")\"");
			}
			resultsOut.println();
		}
		resultsOut.println();
		resultsOut.println();
		resultsOut.println("\"SUMMARY STATISTICS\"");
		resultsOut.println("\"The winner of this table is determined by most winning matches.\"");
		resultsOut.println("\"Ties are broken according to total game wins.\"");
		resultsOut.println("\"Further ties are broken according to total points across all games.\"");
		resultsOut.println(",\"Match Wins\",\"Total Game Wins\",\"Total Points\"");
		for (int i = 0; i < competitors.length; i++) {
			resultsOut.println(i + "-" + playerIdentifiers[i] + "," + wins[i] + "," + totalWins[i] + "," + points[i]);
		}
		resultsOut.close();

		System.out.println("Time taken to calculate results: " + (System.currentTimeMillis() - startTime) + "ms");
	}

	/**
	 * Run <code>NUM_GAMES</code> games of MancalaPlayers with the given name prefixes, 
	 * printing a game transcript to the given PrintWriter to an output file.
	 * @param player1 - prefix of first player MancalaPlayer class name.
	 * @param player2 - prefix of second player MancalaPlayer class name.
	 * @param p - PrintWriter for printing a game transcript to a file
	 * @return an int array containing the player 1 games won, player 2 games won,
	 *  player 1 points scored, and player 2 points scored.
	 */
	private static int[] runGames(String player1, String player2, PrintWriter p) {
		try {
			SPPlayer[] players = new SPPlayer[2];
			Class<?> playerClass = Class.forName(player1);
			players[0] = (SPPlayer) playerClass.getDeclaredConstructor().newInstance();
			playerClass = Class.forName(player2);
			players[1] = (SPPlayer) playerClass.getDeclaredConstructor().newInstance();

			int[] scores = new int[4]; // Player 1 games won, player 2 games won, player 1 points scored, player 2 points scored
			//long[] times = new long[2];

			for (int g = 0; g < NUM_GAMES; g++) { // best of NUM_GAMES
				System.out.println("Game " + (g + 1) + " started");
				//boolean ranOutClock = false;
				//long[] playerMillisRemaining = {MILLISECONDS_PER_GAME/2L, MILLISECONDS_PER_GAME/2L};

				// Create a clock
				//Stopwatch clock = new Stopwatch();
				//long timeTaken;

				// Create a node with a random FairKalah board initial state
				SPState node = new SPState();
				p.println(node);

				// While game is on...
				int move;
				String winner = "DRAW";
				while (!node.isGameOver()) {
					int currentPlayer = node.playerTurn;

					// Request move from current player
					// clock.reset();
					// clock.start();
					move = players[node.playerTurn].getAction((SPState) node.clone()); // , playerMillisRemaining[node.player]);
					// timeTaken = clock.stop();

					// Deduct time taken
					//playerMillisRemaining[node.playerTurn] -= timeTaken;
					// if (playerMillisRemaining[node.playerTurn] < 0) {
					// 	ranOutClock = true;
					// 	if (node.playerTurn == 0) {
					// 		p.println("Player 1 game timer expired.");
					// 		scores[3] += 48;
					// 		winner = "PLAYER 2 WINS";
					// 	} else {
					// 		p.println("Player 2 game timer expired.");
					// 		scores[2] += 48;
					// 		winner = "PLAYER 1 WINS";
					// 	}
					// 	break;
					// }

					try {
						// Update game state
                        ArrayList<SPAction> legalActions = node.getLegalActions();
                        if (move < 0 || move >= legalActions.size()) {
                            throw new IllegalArgumentException("Player " + ((currentPlayer == 0) ? "1" : "2") + " chose an invalid action index: " + move);
                        }
						SPAction action = legalActions.get(move);
                        p.println("> " + action);
                        p.println();
                        action.take();
                        
						// Display Progress
						p.println(node);
					}
					catch (Exception e) {
						p.println("ERROR: Player " + ((currentPlayer == 0) ? "1" : "2") + " makes an ILLEGAL MOVE and forfeits the game.");
						winner = (currentPlayer == 0) ? "PLAYER 2 WINS" : "PLAYER 1 WINS";
						break;
					}
				}

				// Display winner and statistics
				if (node.isGameOver()) {
					if (node.playerPoints[0] > node.playerPoints[1])
						winner = "PLAYER 1 WINS";
					else
						if (node.playerPoints[0] < node.playerPoints[1])
							winner = "PLAYER 2 WINS";
						else
							winner = "DRAW";
				}
				// p.println("Time Taken (ms): ");
				// long t = MILLISECONDS_PER_GAME / 2L - playerMillisRemaining[GameNode.MAX];
				// times[0] += t;
				// p.println("Player 1: " + t);
				// t = MILLISECONDS_PER_GAME / 2L - playerMillisRemaining[GameNode.MIN];
				// times[1] += t;
				// p.println("Player 2: " + t);
				p.println(winner);
				if (winner.equals("PLAYER 1 WINS"))
					scores[0]++;
				if (winner.equals("PLAYER 2 WINS"))
					scores[1]++;
				// if (!ranOutClock) {
                scores[2] += node.playerPoints[0];
                scores[3] += node.playerPoints[1];
				// }
				p.println("-----------------------------------------------------------------------------------");
			}
			p.println("Player 1 wins: "+scores[0]);
			p.println("Player 2 wins: "+scores[1]);
			p.println("Player 1 points won: "+scores[2]);
			p.println("Player 2 points won: "+scores[3]);
			// p.println("Player 1 total time used: "+times[0]);
			// p.println("Player 2 total time used: "+times[1]);
			p.println("Player 1 name: " + player1);
			p.println("Player 2 name: " + player2);
			p.flush();
			return scores;

		}  catch (Exception e) {
			e.printStackTrace();
			return null;
		} 
	}
} // Tournament
