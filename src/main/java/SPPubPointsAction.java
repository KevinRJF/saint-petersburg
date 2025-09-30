import java.util.Arrays;

public class SPPubPointsAction extends SPAction {

	public final int points; // The number of points to purchase

	public SPPubPointsAction(SPState state, int points) {
		super(state);
		this.points = points;
	}

	@Override
	public SPState take(SPState state) {
		state.playerPoints[state.playerTurn] += points; // Add points to the player's score
		state.playerRubles[state.playerTurn] -= 2 * points; // Deduct the cost of points

		// If there is another player in the phase turn order with a pub, continue to the next player, otherwise end the phase.
		boolean hasNextPlayerWithPub = false;
		int playerOffset = (state.playerTurn + state.numPlayers - state.startingPlayer[SPState.BUILDING]) % state.numPlayers;
		for (int offset = playerOffset + 1; offset < state.numPlayers; offset++) {
			int possibleNextPubPlayer = (state.startingPlayer[SPState.BUILDING] + offset) % state.numPlayers;
			if (state.playerBuildings.get(possibleNextPubPlayer).stream().anyMatch(c -> c.name.equals("Pub"))) {
				hasNextPlayerWithPub = true;
				state.playerTurn = possibleNextPubPlayer; // Set the next player with a Pub
				break;
			}
		}
		if (!hasNextPlayerWithPub) { // end the phase if no other player has a Pub
			Arrays.fill(state.usedObservatories, 0); // Reset observatory usage for next round
			state.phase = SPState.ARISTOCRAT; // move to aristocrat phase
			refillTopRow(state.aristocratDeck); // refill upper card row with aristocrats
			state.playerTurn = state.startingPlayer[state.phase]; // reset player turn to starting player for aristocrat phase
		}
		return state;
	}

	@Override
	public String toString() {
		return points == 0 
			? String.format("Player %d opts not to use the pub.", player + 1) 
			: String.format("Player %d buys %d point%s for %d ruble%s with the Pub.", player + 1, points, points > 1 ? "s" : "", 2 * points, 2 * points > 1 ? "s" : "");
	}
}
