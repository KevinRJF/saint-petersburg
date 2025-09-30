public class SPFeatureInteractionTerm extends SPFeature {
    private SPFeature feature1;
    private SPFeature feature2;

    public SPFeatureInteractionTerm(SPFeature feature1, SPFeature feature2) {
        super(feature1.getName() + "_x_" + feature2.getName(),
            String.format("(%s) * (%s)", feature1.getDescription(), feature2.getDescription()));
        this.feature1 = feature1;
        this.feature2 = feature2;
    }

    @Override
    public Object getValue(SPState state) {
        Object value1 = feature1.getValue(state);
        Object value2 = feature2.getValue(state);
        if (value1 instanceof Number && value2 instanceof Number) {
            return ((Number) value1).doubleValue() * ((Number) value2).doubleValue();
        }
        return null;
    }
}
