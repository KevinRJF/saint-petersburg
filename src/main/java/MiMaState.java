import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MiMaState extends SPState{

	public MiMaState(int numPlayers) {
		super (numPlayers);
	}
	
	@Override
	public ArrayList<SPAction> getLegalActions() {
		ArrayList<SPAction> legalActions = new ArrayList<>();
		if (isGameOver()) {
			return legalActions; // No actions allowed in the end phase
		}
		
		boolean almostOver = workerDeck.size() <= 5 || buildingDeck.size() <= 5 || 
				aristocratDeck.size() <= 5 || tradingDeck.size() <= 5;
		if(almostOver) {
			ArrayList<SPCard> currHand = playerHands.get(playerTurn);
			for (SPCard card : currHand) {
				int currRubles = playerRubles[playerTurn];
				int cost = card.cost;
				if (cost <= currRubles) {
					ArrayList<SPCard> cardDestination = null;
					if (card.isWorker) {
						cardDestination = playerWorkers.get(playerTurn);
					} else if (card.isBuilding) {
						cardDestination = playerBuildings.get(playerTurn);
					} else if (card.isAristocrat) {
						cardDestination = playerAristocrats.get(playerTurn);
					}
					SPBuyAction buyAction = new SPBuyAction(this, card, currHand, cardDestination, cost);
					legalActions.add(buyAction);
				}
			}
			
			if (legalActions.size() > 0) {
				return legalActions;
			}
		}

		// Determine if the player has room in their hand for another card
		// Check if the player has a Warehouse building card
		boolean hasWarehouse = playerBuildings.get(playerTurn).stream().anyMatch(b -> b.name.equals("Warehouse"));
		boolean hasRoomInHand = (hasWarehouse ? 4 : 3) - playerHands.get(playerTurn).size() > 0; // Check if the player has room in hand

		// If the player has bought the "Carpenter Workshop", the building costs are reduced by 1
		boolean hasBuildingDiscount = false;
		if (playerBuildings.get(playerTurn).stream()
				.anyMatch(c -> c.name.equals("Carpenter Workshop"))) {
			hasBuildingDiscount = true;
		}

		// If the player has bought the "Gold Smelter", the aristocrat costs are reduced by 1
		boolean hasAristocratDiscount = false;
		if (playerBuildings.get(playerTurn).stream()
				.anyMatch(c -> c.name.equals("Gold Smelter"))) {
			hasAristocratDiscount = true;
		}

		// Determine if the player has an unused observatory
		int numUnusedObservatories = (int) playerBuildings.get(playerTurn).stream()
				.filter(observatory -> observatory.name.equals("Observatory")).count();
		numUnusedObservatories -= usedObservatories[playerTurn]; // Subtract the number of observatories used this round

		// If they have observed a pile, they must choose what to do with the observed card.
		if (observedCard != null) {
			// Discard the observed card
			legalActions.add(new SPDiscardAction(this));
			// If the player has room in hand, they can add the observed card to their hand
			if (hasRoomInHand && !almostOver) {
				legalActions.add(new SPAddToHandAction(this, observedCard, null)); // null indicates the card is not from a row
			}
			// If the player can afford to buy the observed card, add buy action(s)
			int rubles = playerRubles[playerTurn];
			if (rubles > 1) { // all buys require at least 1 ruble) {
				SPCard card = observedCard; // The card being observed
				int cost = card.cost;
				if (card.isBuilding && hasBuildingDiscount) {
					cost--; // Building costs are reduced by 1 if the player has the Carpenter Workshop
				}
				if (card.isAristocrat && hasAristocratDiscount) {
					cost--; // Aristocrat costs are reduced by 1 if the player has the Gold Smelter
				}

				// Determine the destination for the card
				ArrayList<SPCard> cardDestination = null;
				if (card.isWorker) {
					cardDestination = playerWorkers.get(playerTurn);
				} else if (card.isBuilding) {
					cardDestination = playerBuildings.get(playerTurn);
				} else if (card.isAristocrat) {
					cardDestination = playerAristocrats.get(playerTurn);
				}
				// Reduce the cost by 1 for each same card in the destination
				for (SPCard c : cardDestination) {
					if (c == card) {
						cost--;
					}
				}

				// If the card is not a trading card, create a buy action if affordable
				if (!card.isTrading) {
					if (cost < 1) {
						cost = 1; // Ensure the cost is not less than 1
					}
					if (rubles >= cost) {
						// Create a buy action for the card
						SPBuyAction buyAction = new SPBuyAction(this, card, null, cardDestination, cost);
						legalActions.add(buyAction);
					}
				} // end observed non-trading card handling 
				else { // If the card is a trading card, create a buy action(s) with replaced card(s)
					// Compute the replaceable cards with associated costs
					Map<SPCard, Integer> replaceableCards = new HashMap<>();
					for (SPCard c : cardDestination) {
						if (c.isTrading) {
							continue; // cannot replace trading cards with trading cards
						}
						int replaceCost = cost - c.cost; // Cost of replacing the card
						if (c.name.equals("Observatory")) {
							if (numUnusedObservatories < 1) {
								continue; // Cannot replace an Observatory if none are left unused
							}
						}
						if (card.name.endsWith("Village")) {
							replaceCost -= 4; // Potjomkin's/Potemkin Village credits 6 while only costing 2
						}
						if (replaceCost < 1) {
							replaceCost = 1; // Ensure the cost is not less than 1
						}
						if (rubles >= replaceCost) {
							replaceableCards.put(c, replaceCost); // Add the card and its cost to the map
						}
					}
					// Create a buy action for each replaceable card
					for (Map.Entry<SPCard, Integer> entry : replaceableCards.entrySet()) {
						SPCard replacedCard = entry.getKey();
						int replaceCost = entry.getValue();
						SPBuyAction buyAction = new SPBuyAction(this, card, null, cardDestination, replaceCost, replacedCard);
						legalActions.add(buyAction);
					}
				} // end observed trading card handling
			} // end has at least 1 ruble to buy observed card
		} // end observed card handling
		else if (phase == PUB_ACTION) {
			// Determine the number of pubs the player has
			int numPubs = (int) playerBuildings.get(playerTurn).stream().filter(c -> c.name.equals("Pub")).count();
			int maxPoints = numPubs * 5; // Each Pub can purchase up to 5 points at 2 rubles/point
			int rubles = playerRubles[playerTurn];
			for (int points = 0; points <= maxPoints && 2 * points <= rubles; points++) {
				legalActions.add(new SPPubPointsAction(this, points));
			}
		}
		// Otherwise, is it a normal pass/observe/buy/add-to-hand decision.
		else {
			
			// Add pass action
			legalActions.add(new SPPassAction(this));
			
			// Add buy actions
			int rubles = playerRubles[playerTurn];

			if (rubles > 0) { // all buys require at least 1 ruble

				ArrayList<ArrayList<SPCard>> cardSources = new ArrayList<>();
				cardSources.add(playerHands.get(playerTurn)); // Player's hand
				cardSources.add(upperCardRow);
				cardSources.add(lowerCardRow);

				for (ArrayList<SPCard> cardSource : cardSources) {
					for (SPCard card : cardSource) {
						int cost = card.cost;
						if (cardSource == lowerCardRow) {
							cost--; // Lower row cards cost 1 less
						}
						if (card.isBuilding && hasBuildingDiscount) {
							cost--; // Building costs are reduced by 1 if the player has the Carpenter Workshop
						}
						if (card.isAristocrat && hasAristocratDiscount) {
							cost--; // Aristocrat costs are reduced by 1 if the player has the Gold Smelter
						}
						// Determine the destination for the card
						ArrayList<SPCard> cardDestination = null;
						if (card.isWorker) {
							cardDestination = playerWorkers.get(playerTurn);
						} else if (card.isBuilding) {
							cardDestination = playerBuildings.get(playerTurn);
						} else if (card.isAristocrat) {
							cardDestination = playerAristocrats.get(playerTurn);
						}
						// Reduce the cost by 1 for each same card in the destination
						for (SPCard c : cardDestination) {
							if (c == card) {
								cost--;
							}
						}
						// If the card is not a trading card, create a buy action if affordable
						if (!card.isTrading) {
							if (cost < 1) {
								cost = 1; // Ensure the cost is not less than 1
							}
							if (rubles >= cost) {
								// Create a buy action for the card
								SPBuyAction buyAction = new SPBuyAction(this, card, cardSource, cardDestination, cost);
								legalActions.add(buyAction);
							}
						} else { // If the card is a trading card, create a buy action(s) with replaced card(s)
							// Compute the replaceable cards with associated costs
							Map<SPCard, Integer> replaceableCards = new HashMap<>();
							for (SPCard c : cardDestination) {
								if (c.isTrading) {
									continue; // cannot replace trading cards with trading cards
								}
								int replaceCost = cost - c.cost; // Cost of replacing the card
								if (c.name.equals("Observatory")) {
									if (numUnusedObservatories < 1) {
										continue; // Cannot replace an Observatory if none are left unused
									}
								}
								if (card.name.endsWith("Village")) {
									replaceCost -= 4; // Potjomkin's/Potemkin Village credits 6 while only costing 2
								}
								if (replaceCost < 1) {
									replaceCost = 1; // Ensure the cost is not less than 1
								}
								if (rubles >= replaceCost) {
									replaceableCards.put(c, replaceCost); // Add the card and its cost to the map
								}
							}
							// Create a buy action for each replaceable card
							for (Map.Entry<SPCard, Integer> entry : replaceableCards.entrySet()) {
								SPCard replacedCard = entry.getKey();
								int replaceCost = entry.getValue();
								SPBuyAction buyAction = new SPBuyAction(this, card, cardSource, cardDestination, replaceCost, replacedCard);
								legalActions.add(buyAction);
							}
						} // end trading card handling
					} // end for each card in card source
				} // end for each card source
			} // end buy actions

			// Add observe actions
			if (numUnusedObservatories > 0) { // Only add observe actions if the player has unused observatories
				if (workerDeck.size() > 1) {
					legalActions.add(new SPObserveDeckAction(this, 0));
				}
				if (buildingDeck.size() > 1) {
					legalActions.add(new SPObserveDeckAction(this, 1));
				}
				if (aristocratDeck.size() > 1) {
					legalActions.add(new SPObserveDeckAction(this, 2));
				}
				if (tradingDeck.size() > 1) {
					legalActions.add(new SPObserveDeckAction(this, 3));
				}
			} // end observe actions

			// Add add-to-hand actions

			if (hasRoomInHand && !almostOver) {
				// Add actions to add cards from the upper card row to the player's hand
				for (SPCard card : upperCardRow) {
					legalActions.add(new SPAddToHandAction(this, card, upperCardRow));
				}
				// Add actions to add cards from the lower card row to the player's hand
				for (SPCard card : lowerCardRow) {
					legalActions.add(new SPAddToHandAction(this, card, lowerCardRow));
				}
			}
		} // end else for normal actions
		return legalActions;
	}
	
	@SuppressWarnings("unchecked")
	public MiMaState clone(){
			MiMaState copy = (MiMaState) super.clone(); // Shallow copy
			copy.workerDeck = (ArrayList<SPCard>) workerDeck.clone();
			copy.buildingDeck = (ArrayList<SPCard>) buildingDeck.clone();
			copy.aristocratDeck = (ArrayList<SPCard>) aristocratDeck.clone();
			copy.tradingDeck = (ArrayList<SPCard>) tradingDeck.clone();
			copy.playerHands = new ArrayList<ArrayList<SPCard>>();
			copy.playerWorkers = new ArrayList<ArrayList<SPCard>>();
			copy.playerBuildings = new ArrayList<ArrayList<SPCard>>();
			copy.playerAristocrats = new ArrayList<ArrayList<SPCard>>();
			for (int i = 0; i < playerHands.size(); i++) {
				copy.playerHands.add((ArrayList<SPCard>) playerHands.get(i).clone());
				copy.playerWorkers.add((ArrayList<SPCard>) playerWorkers.get(i).clone());
				copy.playerBuildings.add((ArrayList<SPCard>) playerBuildings.get(i).clone());
				copy.playerAristocrats.add((ArrayList<SPCard>) playerAristocrats.get(i).clone());
			}
			copy.upperCardRow = (ArrayList<SPCard>) upperCardRow.clone();
			copy.lowerCardRow = (ArrayList<SPCard>) lowerCardRow.clone();
			copy.discardPile = (ArrayList<SPCard>) discardPile.clone();
			copy.playerRubles = playerRubles.clone();
			copy.playerPoints = playerPoints.clone();
			copy.playerPassed = playerPassed.clone();
			copy.startingPlayer = startingPlayer.clone();
			copy.usedObservatories = usedObservatories.clone();
			copy.isWinner = isWinner.clone();
			return copy;
	}
	
	void initialize() {
		// Initialize decks with cards from SPCard class
		for (SPCard card : SPCard.ALL_CARDS) {
			if (card.isTrading) {
				for (int i = 0; i < card.quantity; i++) {
					tradingDeck.add(card);
				}
			} else if (card.isWorker) {
				for (int i = 0; i < card.quantity; i++) {
					workerDeck.add(card);
				}
			} else if (card.isBuilding) {
				for (int i = 0; i < card.quantity; i++) {
					buildingDeck.add(card);
				}
			} else if (card.isAristocrat) {
				for (int i = 0; i < card.quantity; i++) {
					aristocratDeck.add(card);
				}
			}
		}
		// Shuffle decks
		Collections.shuffle(workerDeck);
		Collections.shuffle(buildingDeck);
		Collections.shuffle(aristocratDeck);	
		Collections.shuffle(tradingDeck);
		// Initial rubles for each player
		Arrays.fill(playerRubles, INITIAL_RUBLES);
		// Initialize upper card row
		int initialWorkers = 2 * numPlayers;
		for (int i = 0; i < initialWorkers; i++) {
			upperCardRow.add(workerDeck.remove(workerDeck.size() - 1));
		}
		// Initial starting roles for each player
		ArrayList<Integer> startingPlayersList = new ArrayList<>();
		for (int i = 0; i < NUM_DECKS; i++) {
			startingPlayersList.add(i % numPlayers);
		}
		Collections.shuffle(startingPlayersList);
		for (int i = 0; i < NUM_DECKS; i++) {
			startingPlayer[i] = startingPlayersList.get(i);
		}
		playerTurn = startingPlayer[WORKER]; // Set the player to start the worker phase of the first round
		// Initialize player hands, workers, buildings, and aristocrats
		for (int i = 0; i < numPlayers; i++) {
			playerHands.add(new ArrayList<>());
			playerWorkers.add(new ArrayList<>());
			playerBuildings.add(new ArrayList<>());
			playerAristocrats.add(new ArrayList<>());
		}
	}
	
	private <T> String listToStringNoBrackets(List<T> list) {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < list.size(); i++) {
			sb.append(list.get(i).toString());
			if (i < list.size() - 1) {
				sb.append(", ");
			}
		}
		return sb.toString();
	}
}
	


