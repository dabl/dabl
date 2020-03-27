from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC

enable_hist_gradient_boosting


def get_fast_classifiers(n_classes):
    """Get a list of very fast classifiers.

    Parameters
    ----------
    n_classes : int
        Number of classes in the dataset. Used to decide on the complexity
        of some of the classifiers.


    Returns
    -------
    fast_classifiers : list of sklearn estimators
        List of classification models that can be fitted and evaluated very
        quickly.
    """
    return [
        # These are sorted by approximate speed
        DummyClassifier(strategy="prior"),
        GaussianNB(),
        make_pipeline(MinMaxScaler(), MultinomialNB()),
        DecisionTreeClassifier(max_depth=1, class_weight="balanced"),
        DecisionTreeClassifier(max_depth=max(5, n_classes),
                               class_weight="balanced"),
        DecisionTreeClassifier(class_weight="balanced",
                               min_impurity_decrease=.01),
        LogisticRegression(C=.1, solver='lbfgs', multi_class='auto',
                           class_weight='balanced', max_iter=1000),
        # FIXME Add warm starting here?
        LogisticRegression(C=1, solver='lbfgs', multi_class='auto',
                           class_weight='balanced', max_iter=1000)
    ]


def get_fast_regressors():
    """Get a list of very fast regressors.

    Returns
    -------
    fast_regressors : list of sklearn estimators
        List of regression models that can be fitted and evaluated very
        quickly.
    """
    return [
        DummyRegressor(),
        DecisionTreeRegressor(max_depth=1),
        DecisionTreeRegressor(max_depth=5),
        Ridge(alpha=10),
        Lasso(alpha=10)]


def get_any_classifiers(portfolio='baseline'):
    """Return a portfolio of classifiers.

    Returns
    -------
    classifiers : list of sklearn estimators
        List of classification models.
    """
    baseline = [
        LogisticRegression(C=1, solver='lbfgs', multi_class='multinomial'),
        LogisticRegression(C=10, solver='lbfgs', multi_class='multinomial'),
        LogisticRegression(C=.1, solver='lbfgs', multi_class='multinomial'),
        RandomForestClassifier(max_features=None, n_estimators=100),
        RandomForestClassifier(max_features='sqrt', n_estimators=100),
        RandomForestClassifier(max_features='log2', n_estimators=100),
        SVC(C=1, gamma=0.03, kernel='rbf'),
        SVC(C=1, gamma='scale', kernel='rbf'),
        HistGradientBoostingClassifier()]

    svc = [
        SVC(C=131.79955149814975, coef0=0.0, degree=3, 
            gamma=0.005449601498518041, kernel='rbf', probability=False),
        SVC(C=6.5031187555491305, coef0=0.3569381453006406, degree=4, 
            gamma=0.15364377981481867, kernel='rbf', probability=False),
        SVC(C=2.5918689981661567, coef0=0.3186996400686849, degree=3, 
            gamma=0.0016271844595562733, kernel='rbf', probability=False),
        SVC(C=8.537098039116069, coef0=0.0, degree=3, 
            gamma=0.014430579841442782, kernel='rbf', probability=False),
        SVC(C=33.252648089739836, coef0=0.0, degree=3, 
            gamma=2.1212339071044592, kernel='rbf', probability=False),
        SVC(C=55762.3529353618, coef0=-0.8056114085510306, degree=3, 
            gamma=3.187772482265977e-05, kernel='sigmoid', probability=False),
        SVC(C=149.07622270551335, coef0=0.0, degree=3, 
            gamma=0.05610768111553853, kernel='rbf', probability=False),
        SVC(C=0.38559962233936546, coef0=0.08722972305625087, degree=3, 
            gamma=7.521110541330819, kernel='rbf', probability=False),
        SVC(C=34.18479740302528, coef0=0.465809282171058, degree=3, 
            gamma=0.025017141595224057, kernel='rbf', probability=False),
        SVC(C=0.42681252219264904, coef0=0.23495235580748663, degree=4, 
            gamma=0.0419665675168468, kernel='rbf', probability=False),
        SVC(C=0.19056746772632044, coef0=0.3565163593285343, degree=5, 
            gamma=0.015503770572916192, kernel='sigmoid', probability=False),
        SVC(C=81.8664880584341, coef0=0.1339044447397313, degree=4, 
            gamma=0.6339071538529285, kernel='rbf', probability=False),
        SVC(C=1.799125831143992, coef0=0.7926565732345652, degree=3, 
            gamma=0.01858955180141993, kernel='poly', probability=False),
        SVC(C=100000.0, coef0=0.0, degree=3, 
            gamma=0.00241115807647, kernel='rbf', probability=False),
        SVC(C=4.104647380564808, coef0=-0.724712336449596, degree=5, 
            gamma=0.2926981232494074, kernel='rbf', probability=False),
        SVC(C=0.14135351008197172, coef0=-0.6091263079805762, degree=5, 
            gamma=3.1086101104977364e-05, kernel='poly', probability=False),
        SVC(C=8.973055290432324, coef0=-0.5701344387842959, degree=5, 
            gamma=0.007654592599531776, kernel='rbf', probability=False),
        SVC(C=53212.0921773362, coef0=-0.20185133000171707, degree=4, 
            gamma=7.496099998815888e-05, kernel='rbf', probability=False),
        SVC(C=1622.763547526942, coef0=0.2316204140676945, degree=3, 
            gamma=0.042609827284645976, kernel='rbf', probability=False),
        SVC(C=9.62953169042274, coef0=0.0, degree=3, 
            gamma=0.037086661651597304, kernel='rbf', probability=False),
        SVC(C=6.731333187866344, coef0=0.5722855162334912, degree=4, 
            gamma=0.00026798359141741787, kernel='poly', probability=False),
        SVC(C=32.184662024224764, coef0=0.8359658612498964, degree=2, 
            gamma=0.11650886331310951, kernel='poly', probability=False),
        SVC(C=128.0, coef0=0.0, degree=3, 
            gamma=0.03125, kernel='rbf', probability=False),
        SVC(C=513.6546159384284, coef0=-0.16364828944715204, degree=5, 
            gamma=0.3170962383864124, kernel='rbf', probability=False),
        SVC(C=1.0357760788047117, coef0=0.420858393631554, degree=4, 
            gamma=6.420989750880023, kernel='rbf', probability=False),
        SVC(C=4.523994003904736, coef0=0.20331467465699182, degree=5, 
            gamma=0.033987763661532146, kernel='rbf', probability=False),
        SVC(C=8495.49361658936, coef0=0.0, degree=3, 
            gamma=8.947218821753246e-05, kernel='rbf', probability=False),
        SVC(C=3.081620245614723, coef0=-0.053322514098660845, degree=5, 
            gamma=0.31447231452702795, kernel='poly', probability=False),
        SVC(C=4.6521870291674965, coef0=-0.2115772105392888, degree=4, 
            gamma=0.1618086969775427, kernel='rbf', probability=False),
        SVC(C=6.850412137824285, coef0=0.0, degree=3, 
            gamma=0.05170269991130989, kernel='rbf', probability=False),
        SVC(C=12825.233283804411, coef0=-0.5596962381502244, degree=1, 
            gamma=0.007137693124484331, kernel='rbf', probability=False),
        SVC(C=0.12779439580461893, coef0=-0.007860547843195675, degree=2, 
            gamma=0.011489094638370643, kernel='sigmoid', probability=False)]

    hgb = [
        HistGradientBoostingClassifier(l2_regularization=1e-06,
                                       learning_rate=0.1, loss='auto', 
                                       max_bins=255, max_depth=None,
                                       max_iter=200, max_leaf_nodes=128, 
                                       min_samples_leaf=50),
        HistGradientBoostingClassifier(l2_regularization=1.0,
                                       learning_rate=0.1, loss='auto', 
                                       max_bins=64, max_depth=5,
                                       max_iter=100, max_leaf_nodes=4, 
                                       min_samples_leaf=1),
        HistGradientBoostingClassifier(l2_regularization=1.0,
                                       learning_rate=0.1, loss='auto', 
                                       max_bins=64, max_depth=18,
                                       max_iter=350, max_leaf_nodes=32, 
                                       min_samples_leaf=7),
        HistGradientBoostingClassifier(l2_regularization=1e-07,
                                       learning_rate=0.1, loss='auto', 
                                       max_bins=16, max_depth=19,
                                       max_iter=500, max_leaf_nodes=8, 
                                       min_samples_leaf=27),
        HistGradientBoostingClassifier(l2_regularization=10.0,
                                       learning_rate=0.1, loss='auto', 
                                       max_bins=255, max_depth=16,
                                       max_iter=100, max_leaf_nodes=128, 
                                       min_samples_leaf=8),
        HistGradientBoostingClassifier(l2_regularization=1e-07,
                                       learning_rate=0.1, loss='auto', 
                                       max_bins=255, max_depth=16,
                                       max_iter=350, max_leaf_nodes=128, 
                                       min_samples_leaf=13),
        HistGradientBoostingClassifier(l2_regularization=10.0,
                                       learning_rate=0.1, loss='auto', 
                                       max_bins=8, max_depth=20,
                                       max_iter=150, max_leaf_nodes=4, 
                                       min_samples_leaf=13),
        HistGradientBoostingClassifier(l2_regularization=1e-07,
                                       learning_rate=0.1, loss='auto', 
                                       max_bins=16, max_depth=3,
                                       max_iter=350, max_leaf_nodes=16, 
                                       min_samples_leaf=14),
        HistGradientBoostingClassifier(l2_regularization=10.0,
                                       learning_rate=1.0, loss='auto', 
                                       max_bins=255, max_depth=12,
                                       max_iter=250, max_leaf_nodes=32, 
                                       min_samples_leaf=42),
        HistGradientBoostingClassifier(l2_regularization=0.01,
                                       learning_rate=0.01, loss='auto', 
                                       max_bins=8, max_depth=19,
                                       max_iter=350, max_leaf_nodes=4, 
                                       min_samples_leaf=3),
        HistGradientBoostingClassifier(l2_regularization=0.001,
                                       learning_rate=0.1, loss='auto', 
                                       max_bins=64, max_depth=14,
                                       max_iter=200, max_leaf_nodes=16, 
                                       min_samples_leaf=3),
        HistGradientBoostingClassifier(l2_regularization=0.01,
                                       learning_rate=0.1, loss='auto', 
                                       max_bins=255, max_depth=3,
                                       max_iter=400, max_leaf_nodes=16, 
                                       min_samples_leaf=11),
        HistGradientBoostingClassifier(l2_regularization=1e-05,
                                       learning_rate=0.1, loss='auto', 
                                       max_bins=16, max_depth=None,
                                       max_iter=400, max_leaf_nodes=128, 
                                       min_samples_leaf=48),
        HistGradientBoostingClassifier(l2_regularization=0.001,
                                       learning_rate=0.1, loss='auto', 
                                       max_bins=255, max_depth=18,
                                       max_iter=450, max_leaf_nodes=64, 
                                       min_samples_leaf=9),
        HistGradientBoostingClassifier(l2_regularization=10.0,
                                       learning_rate=0.1, loss='auto', 
                                       max_bins=255, max_depth=20,
                                       max_iter=400, max_leaf_nodes=64, 
                                       min_samples_leaf=5),
        HistGradientBoostingClassifier(l2_regularization=1e-06,
                                       learning_rate=0.1, loss='auto', 
                                       max_bins=255, max_depth=8,
                                       max_iter=500, max_leaf_nodes=4, 
                                       min_samples_leaf=9),
        HistGradientBoostingClassifier(l2_regularization=0.1,
                                       learning_rate=0.1, loss='auto', 
                                       max_bins=16, max_depth=17,
                                       max_iter=50, max_leaf_nodes=4, 
                                       min_samples_leaf=23),
        HistGradientBoostingClassifier(l2_regularization=0.01,
                                       learning_rate=0.1, loss='auto', 
                                       max_bins=16, max_depth=4,
                                       max_iter=450, max_leaf_nodes=64, 
                                       min_samples_leaf=19),
        HistGradientBoostingClassifier(l2_regularization=1.0,
                                       learning_rate=0.1, loss='auto', 
                                       max_bins=16, max_depth=2,
                                       max_iter=300, max_leaf_nodes=16, 
                                       min_samples_leaf=8),
        HistGradientBoostingClassifier(l2_regularization=0.001,
                                       learning_rate=0.01, loss='auto', 
                                       max_bins=64, max_depth=9,
                                       max_iter=450, max_leaf_nodes=32, 
                                       min_samples_leaf=12),
        HistGradientBoostingClassifier(l2_regularization=10.0,
                                       learning_rate=0.1, loss='auto', 
                                       max_bins=255, max_depth=15,
                                       max_iter=200, max_leaf_nodes=64, 
                                       min_samples_leaf=1),
        HistGradientBoostingClassifier(l2_regularization=100.0,
                                       learning_rate=0.1, loss='auto', 
                                       max_bins=64, max_depth=7,
                                       max_iter=350, max_leaf_nodes=8, 
                                       min_samples_leaf=5),
        HistGradientBoostingClassifier(l2_regularization=0.01,
                                       learning_rate=0.1, loss='auto', 
                                       max_bins=128, max_depth=16,
                                       max_iter=200, max_leaf_nodes=32, 
                                       min_samples_leaf=16),
        HistGradientBoostingClassifier(l2_regularization=1.0,
                                       learning_rate=1.0, loss='auto', 
                                       max_bins=16, max_depth=2,
                                       max_iter=150, max_leaf_nodes=32, 
                                       min_samples_leaf=8),
        HistGradientBoostingClassifier(l2_regularization=1.0,
                                       learning_rate=0.1, loss='auto', 
                                       max_bins=128, max_depth=16,
                                       max_iter=100, max_leaf_nodes=8, 
                                       min_samples_leaf=5),
        HistGradientBoostingClassifier(l2_regularization=0.0001,
                                       learning_rate=0.1, loss='auto', 
                                       max_bins=128, max_depth=20,
                                       max_iter=500, max_leaf_nodes=128, 
                                       min_samples_leaf=3),
        HistGradientBoostingClassifier(l2_regularization=0.1,
                                       learning_rate=0.01, loss='auto', 
                                       max_bins=4, max_depth=18,
                                       max_iter=200, max_leaf_nodes=4, 
                                       min_samples_leaf=39),
        HistGradientBoostingClassifier(l2_regularization=0.1,
                                       learning_rate=0.1, loss='auto', 
                                       max_bins=32, max_depth=12,
                                       max_iter=200, max_leaf_nodes=128, 
                                       min_samples_leaf=3),
        HistGradientBoostingClassifier(l2_regularization=1e-05,
                                       learning_rate=0.1, loss='auto', 
                                       max_bins=255, max_depth=16,
                                       max_iter=400, max_leaf_nodes=64, 
                                       min_samples_leaf=10),
        HistGradientBoostingClassifier(l2_regularization=1e-10,
                                       learning_rate=0.1, loss='auto', 
                                       max_bins=64, max_depth=2,
                                       max_iter=100, max_leaf_nodes=4, 
                                       min_samples_leaf=3),
        HistGradientBoostingClassifier(l2_regularization=0.1,
                                       learning_rate=0.1, loss='auto', 
                                       max_bins=255, max_depth=4,
                                       max_iter=350, max_leaf_nodes=128, 
                                       min_samples_leaf=11),
        HistGradientBoostingClassifier(l2_regularization=1e-10,
                                       learning_rate=0.1, loss='auto', 
                                       max_bins=128, max_depth=None,
                                       max_iter=500, max_leaf_nodes=8, 
                                       min_samples_leaf=21)]
    
    mixed = [
        HistGradientBoostingClassifier(l2_regularization=1e-06,
                                       learning_rate=0.1, loss='auto',
                                       max_bins=255, max_depth=None,
                                       max_iter=200, max_leaf_nodes=128,
                                       min_samples_leaf=50),

        SVC(C=52.368035023140784, coef0=0.0, degree=3,
            gamma=0.008051730038808798, kernel='rbf', probability=False),

        HistGradientBoostingClassifier(l2_regularization=10.0,
                                       learning_rate=0.1, loss='auto',
                                       max_bins=255, max_depth=15,
                                       max_iter=200, max_leaf_nodes=64,
                                       min_samples_leaf=1),

        SVC(C=1.799125831143992, coef0=0.7926565732345652, degree=3,
            gamma=0.01858955180141993, kernel='poly', probability=False),

        HistGradientBoostingClassifier(l2_regularization=0.001,
                                       learning_rate=0.1, loss='auto',
                                       max_bins=255, max_depth=18,
                                       max_iter=450, max_leaf_nodes=64,
                                       min_samples_leaf=9),

        SVC(C=2854.2115055717222, coef0=0.9097801350305286, degree=2,
            gamma=0.0594124208513135, kernel='rbf', probability=False),

        HistGradientBoostingClassifier(l2_regularization=100.0,
                                       learning_rate=0.1, loss='auto',
                                       max_bins=255, max_depth=17,
                                       max_iter=200, max_leaf_nodes=128,
                                       min_samples_leaf=37),

        SVC(C=0.19056746772632044, coef0=0.3565163593285343, degree=5,
            gamma=0.015503770572916192, kernel='sigmoid', probability=False),

        HistGradientBoostingClassifier(l2_regularization=1e-05,
                                       learning_rate=0.1, loss='auto',
                                       max_bins=32, max_depth=10,
                                       max_iter=350, max_leaf_nodes=8,
                                       min_samples_leaf=2),

        SVC(C=128.0, coef0=0.0, degree=3,
            gamma=0.03125, kernel='rbf', probability=False),

        HistGradientBoostingClassifier(l2_regularization=1.0,
                                       learning_rate=0.1, loss='auto',
                                       max_bins=64, max_depth=18,
                                       max_iter=350, max_leaf_nodes=32,
                                       min_samples_leaf=7),

        SVC(C=4297.397059178814, coef0=0.013499300518863278, degree=5,
            gamma=0.1820082271280914, kernel='rbf', probability=False),

        HistGradientBoostingClassifier(l2_regularization=1e-10,
                                       learning_rate=0.01, loss='auto',
                                       max_bins=64, max_depth=6,
                                       max_iter=500, max_leaf_nodes=4,
                                       min_samples_leaf=19),

        HistGradientBoostingClassifier(l2_regularization=10.0,
                                       learning_rate=0.1, loss='auto',
                                       max_bins=32, max_depth=11,
                                       max_iter=50, max_leaf_nodes=64,
                                       min_samples_leaf=1),

        HistGradientBoostingClassifier(l2_regularization=0.01,
                                       learning_rate=0.1, loss='auto',
                                       max_bins=255, max_depth=5,
                                       max_iter=350, max_leaf_nodes=8,
                                       min_samples_leaf=4),

        HistGradientBoostingClassifier(l2_regularization=1.0,
                                       learning_rate=0.1, loss='auto',
                                       max_bins=255, max_depth=10,
                                       max_iter=450, max_leaf_nodes=64,
                                       min_samples_leaf=12),

        SVC(C=0.6924057373577415, coef0=0.46594279667222027, degree=2,
            gamma=0.027501589087229522, kernel='poly', probability=False),

        HistGradientBoostingClassifier(l2_regularization=0.001,
                                       learning_rate=0.1, loss='auto',
                                       max_bins=64, max_depth=14,
                                       max_iter=200, max_leaf_nodes=16,
                                       min_samples_leaf=3),

        SVC(C=100000.0, coef0=0.0, degree=3,
            gamma=0.00194197897644, kernel='rbf', probability=False),

        SVC(C=0.12779439580461893, coef0=-0.007860547843195675, degree=2,
            gamma=0.011489094638370643, kernel='sigmoid', probability=False),

        HistGradientBoostingClassifier(l2_regularization=0.0001,
                                       learning_rate=0.1, loss='auto',
                                       max_bins=128, max_depth=20,
                                       max_iter=500, max_leaf_nodes=128,
                                       min_samples_leaf=3),

        HistGradientBoostingClassifier(l2_regularization=1e-10,
                                       learning_rate=0.1, loss='auto',
                                       max_bins=128, max_depth=None,
                                       max_iter=500, max_leaf_nodes=8,
                                       min_samples_leaf=21),

        HistGradientBoostingClassifier(l2_regularization=0.001,
                                       learning_rate=0.01, loss='auto',
                                       max_bins=64, max_depth=9,
                                       max_iter=450, max_leaf_nodes=32,
                                       min_samples_leaf=12),

        HistGradientBoostingClassifier(l2_regularization=0.01,
                                       learning_rate=0.1, loss='auto',
                                       max_bins=128, max_depth=16,
                                       max_iter=200, max_leaf_nodes=32,
                                       min_samples_leaf=16),

        HistGradientBoostingClassifier(l2_regularization=1e-09,
                                       learning_rate=0.01, loss='auto',
                                       max_bins=64, max_depth=None,
                                       max_iter=450, max_leaf_nodes=4,
                                       min_samples_leaf=3),

        HistGradientBoostingClassifier(l2_regularization=10.0,
                                       learning_rate=0.1, loss='auto',
                                       max_bins=255, max_depth=20,
                                       max_iter=400, max_leaf_nodes=64,
                                       min_samples_leaf=5),

        HistGradientBoostingClassifier(l2_regularization=1e-06,
                                       learning_rate=0.1, loss='auto',
                                       max_bins=128, max_depth=12,
                                       max_iter=300, max_leaf_nodes=4,
                                       min_samples_leaf=3),

        HistGradientBoostingClassifier(l2_regularization=1e-05,
                                       learning_rate=0.1, loss='auto',
                                       max_bins=16, max_depth=None,
                                       max_iter=400, max_leaf_nodes=128,
                                       min_samples_leaf=48),

        SVC(C=9366.20074943413, coef0=0.0, degree=3,
            gamma=0.0015583202500353984, kernel='rbf', probability=False),

        HistGradientBoostingClassifier(l2_regularization=1e-09,
                                       learning_rate=0.1, loss='auto',
                                       max_bins=8, max_depth=7,
                                       max_iter=150, max_leaf_nodes=128,
                                       min_samples_leaf=27),

        HistGradientBoostingClassifier(l2_regularization=1e-06,
                                       learning_rate=0.1, loss='auto',
                                       max_bins=255, max_depth=15,
                                       max_iter=350, max_leaf_nodes=8,
                                       min_samples_leaf=20),

        HistGradientBoostingClassifier(l2_regularization=0.1,
                                       learning_rate=0.01, loss='auto',
                                       max_bins=4, max_depth=18,
                                       max_iter=200, max_leaf_nodes=4,
                                       min_samples_leaf=39)]
    
    portfolios = {'baseline': baseline,
                 'mixed': mixed,
                 'svc': svc,
                 'hgb': hgb}

    return(portfolios[portfolio])
