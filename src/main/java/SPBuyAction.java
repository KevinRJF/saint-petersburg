import java.util.ArrayList;
/* Represents a card buy action in the game, from the player's hand, upper row, or lower row.
 * Any constructed action must be a legal action in the current game state.
 * Trading card buying is also supported, where replacedCard is the card being replaced.
 */
public class SPBuyAction extends SPAction {

	public SPCard card; // The card to be bought
	public ArrayList<SPCard> cardSource; // Where the card is bought from (e.g., upper row, lower row, hand)
	public ArrayList<SPCard> cardDestination; // Where the card is added (e.g., player's workers, buildings, aristocrats)
	public int cost; // The cost of the card, calculated based on the game state
	public SPCard replacedCard = null; // The card that is replaced, if buying trading card

	SPBuyAction(SPState state, SPCard card, ArrayList<SPCard> cardSource, ArrayList<SPCard> cardDestination, int cost) {
		super(state);
		this.card = card;
		this.cardSource = cardSource;
		this.cardDestination = cardDestination;
		this.cost = cost;
	}

	SPBuyAction(SPState state, SPCard card, ArrayList<SPCard> cardSource, ArrayList<SPCard> cardDestination, int cost, SPCard replacedCard) {
		this(state, card, cardSource, cardDestination, cost);
		this.replacedCard = replacedCard;
	}

	@Override
	public SPState take(SPState state) {
		// Player did not pass
		state.playerPassed[state.playerTurn] = false;

		// Charge the player for the card
		state.playerRubles[state.playerTurn] -= cost;

		// Discard replaced card if relevant
		if (replacedCard != null) {
			if (!cardDestination.remove(replacedCard)) {
				throw new IllegalStateException("Replaced card not found in the destination: " + replacedCard.name);
			}	
		}

		// Remove the card being bought
		if (cardSource == null) {
			state.observedCard = null; // Clear the observed card if it was the one being bought
		}
		else if (!cardSource.remove(card)) {
			throw new IllegalStateException("Card not found in the source: " + card.name);
		}

		// Add the card to the destination
		cardDestination.add(card);

		// Advance the turn to the next player
		state.playerTurn = (state.playerTurn + 1) % state.numPlayers;
		return state;
	}

	@Override
	public String toString() {
		String cardSourceName = "observation";
		if (cardSource == state.upperCardRow) {
			cardSourceName = "the upper card row";
		}
		else if (cardSource == state.lowerCardRow) {
			cardSourceName = "the lower card row";
		}
		else if (cardSource != null) {
			cardSourceName = "their hand";
		}
		String replacingCardInformation = "";
		if (replacedCard != null) {
			replacingCardInformation = String.format(" replacing %s", replacedCard.name);
		}
		return String.format("Player %d buys %s from %s for %d ruble%s%s.", player + 1, card.name, cardSourceName, cost, cost > 1 ? "s" : "", replacingCardInformation);
	}
}
