import java.util.ArrayList;

public class SPAddToHandAction extends SPAction {

	private SPCard card;
	private ArrayList<SPCard> cardRow; // null if adding from observed card

	public SPAddToHandAction(SPState state, SPCard card, ArrayList<SPCard> cardRow) {
		super(state);
		this.card = card;
		this.cardRow = cardRow;
	}

	@Override
    public SPState take(SPState state) {
		// Player did not pass
		state.playerPassed[player] = false;	
		// Remove the card from the card row or observed card
		if (state.observedCard == null) {
			cardRow.remove(card);
		}
		else {
			state.observedCard = null; // Clear the observed card if it was the one being added
		}
		// Add the card to the player's hand
		state.playerHands.get(player).add(card);
		// Advance the turn to the next player
		state.playerTurn = (player + 1) % state.numPlayers;
		return state;
	}

	@Override
	public String toString() {
		return String.format("Player %d adds %s to their hand from %s.", player + 1, card.name, 
			cardRow == null ? "observation" : (cardRow == state.upperCardRow ? "the upper card row" : "the lower card row"));
	}
}
