import java.util.ArrayList;

public abstract class SPAction {

	protected SPState state; // a state for which this is a legal action
	protected int player; // The player who is taking the action

	// Construct an action based on a state for which it is legal
	SPAction(SPState state) {
		this.state = state;
		this.player = state.playerTurn;
	}

	// Take the action on the given state, returning the resulting state.
	abstract public SPState take(SPState state); // assumes no deep cloning of state, returns resulting state
	
	// Take the action on the state for which it is legal, returning the resulting state.
	public SPState take() {
		return take(state);
	}

	// Take the action on the state for which it is legal, 
	// either on a clone of the state or on the state itself depending on the onClone parameter.
	public SPState take(boolean onClone) { // onClone indicates whether or not action will be taken on
		return take(onClone ? (SPState) state.clone() : state);
	}

	public SPCard drawRandomCard(ArrayList<SPCard> deck) {
		if (deck.isEmpty()) {
			return null; // No cards to draw
		}
		int index = (int) (Math.random() * deck.size());
		SPCard card = deck.get(index);
		if (index == deck.size() - 1) {
			deck.remove(index);
		}
		else {
			deck.set(index, deck.get(deck.size() - 1));
			deck.remove(deck.size() - 1);
		}
		return card;
	}

	public void refillTopRow(ArrayList<SPCard> deck) {
		int numCardsToDraw = 8 - state.upperCardRow.size() - state.lowerCardRow.size();
		for (int i = 0; i < numCardsToDraw && !deck.isEmpty(); i++) {
			state.upperCardRow.add(drawRandomCard(deck));
		}
	}
}