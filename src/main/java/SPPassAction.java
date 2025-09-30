import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

public class SPPassAction extends SPAction {

	SPPassAction(SPState state) {
		super(state);
	}

	@Override
	public SPState take(SPState state) {
		state.playerPassed[player] = true; // mark player as having passed

		// Check for end of phase
		boolean allPassed = true;
		for (boolean passed : state.playerPassed) {
			if (!passed) {
				allPassed = false;
				break;
			}
		}
		if (allPassed) {
			// Reset playerPassed for the next phase
			for (int i = 0; i < state.numPlayers; i++) {
				state.playerPassed[i] = false;
			}
			if (state.phase == SPState.WORKER) { // End of worker phase
				// Gain rubles and points for each worker
				for (int i = 0; i < state.numPlayers; i++) {
					for (SPCard card : state.playerWorkers.get(i)) {
						state.playerRubles[i] += card.rubles;
						state.playerPoints[i] += card.points;
					}
				}
				state.phase = SPState.BUILDING; // move to building phase
				refillTopRow(state.buildingDeck); // refill upper card row with buildings
				state.playerTurn = state.startingPlayer[state.phase]; // reset player turn to starting player for building phase
			}
			else if (state.phase == SPState.BUILDING) { // End of building phase
				// Gain rubles and points for each building
				boolean pubActionNext = false; // Flag to check if a Pub is present
				for (int i = 0; i < state.numPlayers; i++) {
					int numObservatories = 0; // Count observatories for each player
					for (SPCard card : state.playerBuildings.get(i)) {
						state.playerRubles[i] += card.rubles;
						state.playerPoints[i] += card.points;
						// If the card is an observatory, count a point if it has not been used this round.
						if (card.name.equals("Observatory")) {
							numObservatories++;
						}
						else if (card.name.startsWith("Mariinskij")) {
							// Score one point per aristocrat card
							state.playerPoints[i] += state.playerAristocrats.get(i).size();
						}
						// If the card is a Pub, there is a Pub pseudo-phase next.
						if (card.name.equals("Pub")) {
							pubActionNext = true;
						}
					}
					// Give a point for each observatory that has not been used this round.
					state.playerPoints[i] += numObservatories - state.usedObservatories[i];
					state.usedObservatories[i] = 0; // Reset observatory usage for next round
				}
				if (pubActionNext) {
					state.phase = SPState.PUB_ACTION; // move to pub phase
					// Find the first player with a Pub, starting with the building phase starting player
					int buildingStartPlayer = state.startingPlayer[SPState.BUILDING];
					for (int i = 0; i < state.numPlayers; i++) {
						int playerIndex = (buildingStartPlayer + i) % state.numPlayers;
						if (state.playerBuildings.get(playerIndex).stream().anyMatch(card -> card.name.equals("Pub"))) {
							state.playerTurn = playerIndex;
							break;
						}
					}
				}
				else { // No Pub, move to aristocrat phase
					Arrays.fill(state.usedObservatories, 0); // Reset observatory usage for next round
					state.phase = SPState.ARISTOCRAT; // move to aristocrat phase
					refillTopRow(state.aristocratDeck); // refill upper card row with aristocrats
					state.playerTurn = state.startingPlayer[state.phase]; // reset player turn to starting player for aristocrat phase
				}
			}
			else if (state.phase == SPState.ARISTOCRAT) { // End of aristocrat phase
				// Gain rubles and points for each aristocrat
				for (int i = 0; i < state.numPlayers; i++) {
					for (SPCard card : state.playerAristocrats.get(i)) {
						state.playerRubles[i] += card.rubles;
						state.playerPoints[i] += card.points;
						if (card.name.equals("Tax Man")) {
							// Gain 1 ruble for each worker card owned by the player
							state.playerRubles[i] += state.playerWorkers.get(i).size();
						}
					}
				}
				state.phase = SPState.TRADING; // move to trading phase
				refillTopRow(state.tradingDeck);
				state.playerTurn = state.startingPlayer[state.phase]; // reset player turn to starting player for trading phase
			}
			else if (state.phase == SPState.TRADING) { // End of trading phase
				state.round++; // increment round
				// Check for end of game
				boolean gameOver = false;
				// If any deck is empty, the game ends
				if (state.workerDeck.isEmpty() || state.buildingDeck.isEmpty() || 
					state.aristocratDeck.isEmpty() || state.tradingDeck.isEmpty()) {
					gameOver = true;
				}
				if (gameOver) { // game over scoring and winner determination
					state.phase = SPState.END; // move to end phase
					// Score unique aristocrats
					for (int i = 0; i < state.numPlayers; i++) {
						Set<SPCard> uniqueAristocrats = new HashSet<>(state.playerAristocrats.get(i));
						int numUniqueAristocrats = Math.min(10, uniqueAristocrats.size());
						state.playerPoints[i] += SPState.UNIQUE_ARISTOCRAT_BONUS_POINTS.get(numUniqueAristocrats);
					}
					// Score remaining rubles at 1 point per 10 rubles
					for (int i = 0; i < state.numPlayers; i++) {
						state.playerPoints[i] += state.playerRubles[i] / 10;
					}
					// Subtract points for remaining cards in hand at -5 per card
					for (int i = 0; i < state.numPlayers; i++) {
						state.playerPoints[i] -= 5 * state.playerHands.get(i).size();
					}
					// Determine the winner
					int maxPoints = Integer.MIN_VALUE;
					for (int i = 0; i < state.numPlayers; i++) {
						if (state.playerPoints[i] > maxPoints) {
							maxPoints = state.playerPoints[i];
						}
					}
					// Compute the max rubles among max points players in case of a tie
					int maxRubles = Integer.MIN_VALUE;
					for (int i = 0; i < state.numPlayers; i++) {
						if (state.playerPoints[i] == maxPoints) {
							if (state.playerRubles[i] > maxRubles) {
								maxRubles = state.playerRubles[i];
							}
						}
					}
					// Mark the winners
					for (int i = 0; i < state.numPlayers; i++) {
						if (state.playerPoints[i] == maxPoints && state.playerRubles[i] == maxRubles) {
							state.isWinner[i] = true;
						}
					}	
				} // game over
				else { // Not game over, continue to next round
					// Discard all lower row cards
					state.discardPile.addAll(state.lowerCardRow);
					// Clear lower row
					state.lowerCardRow.clear();
					// Move all upper row cards to the lower row
					state.lowerCardRow.addAll(state.upperCardRow);
					state.upperCardRow.clear();
					// Refill upper row with new cards from the decks
					refillTopRow(state.workerDeck);
					// rotate starting players for phases clockwise
					for (int i = 0; i < SPState.NUM_DECKS; i++) {
						state.startingPlayer[i] = (state.startingPlayer[i] + 1) % state.numPlayers;
					}
					state.phase = SPState.WORKER; // move to worker phase
					state.playerTurn = state.startingPlayer[state.phase]; // reset player turn to starting player for worker phase
				} // next round
			} // end of trading phase
		} // end all players passed
		else { // Not all players passed, continue current phase
			state.playerTurn = (state.playerTurn + 1) % state.numPlayers; // move to next player
		}
		return state;
	}

	@Override
	public String toString() {
		return String.format("Player %d passes.", player + 1);
	}
}
