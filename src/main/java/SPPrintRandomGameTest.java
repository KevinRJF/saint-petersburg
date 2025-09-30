/* SPPrintRandomGameTest - Tests the game logic and printing functionality with a randomly-played game. */

import java.util.ArrayList;

public class SPPrintRandomGameTest {
	public static void main(String[] args) {
		// Create a new game state
		SPState state = new SPState();
		// Print the initial game state
		System.out.println(state);
		while (!state.isGameOver()) {
			// Get the list of legal actions for the current state
			ArrayList<SPAction> legalActions = state.getLegalActions();
			// Randomly select one of the legal actions
			SPAction action = legalActions.get((int) (Math.random() * legalActions.size()));
			// Print the selected action
			System.out.println(">" + action + "\n");
			// Take the action and update the game state
			state = action.take(state);
			// Print the updated state
			System.out.println(state);
		}
	}
}
