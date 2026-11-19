"""Post-forward diagnostics; ordinary SSC loss and diagnostic masks remain distinct."""
import numpy as np

from scripts.adaptation_ssc_suite.common import require
from scripts.adaptation_tc_windows.metrics import describe

METRICS = ('adjacent_js_nats', 'true_class_concentration', 'true_class_mean_probability',
           'window_accuracy', 'top_class_switch_fraction', 'window_mean_supervised_ce', 'window_supervised_accuracy')


def windows(logits, probabilities, target, support, widths, burn_in=10):
    """Mean frame-softmax diagnostics plus CE on mean raw membrane outputs."""
    z, p = np.asarray(logits, dtype=np.float64), np.asarray(probabilities, dtype=np.float64)
    require(z.ndim == 2 and z.shape == p.shape and np.isfinite(z).all(), 'Invalid frame membrane outputs')
    result = []
    for width in widths:
        row = dict(window_frames=width, **describe(p, target, support, width, burn_in))
        row.update(window_mean_supervised_ce=None, window_supervised_accuracy=None,
                   window_supervised_ce=[], window_supervised_predicted_class=[], adjacent_js_values=[])
        count = row['valid_windows']
        if count >= 2:
            frame_prob = p[burn_in:burn_in+count*width].reshape(count, width, -1).mean(1)
            a, b = frame_prob[:-1], frame_prob[1:]
            mid = (a+b)/2
            tiny = np.finfo(np.float64).tiny
            kl = lambda left, right: np.sum(left*(np.log(np.maximum(left, tiny))-np.log(np.maximum(right, tiny))), axis=1)
            js = np.maximum(0., .5*(kl(a, mid)+kl(b, mid)))
            membrane = z[burn_in:burn_in+count*width].reshape(count, width, -1).mean(1)
            maximum = membrane.max(1)
            ce = maximum+np.log(np.exp(membrane-maximum[:, None]).sum(1))-membrane[:, target]
            require(np.isfinite(ce).all() and np.all(ce >= -1e-12), 'Nonfinite/negative window CE')
            pred = membrane.argmax(1)
            row.update(window_mean_supervised_ce=float(ce.mean()), window_supervised_accuracy=float((pred == target).mean()),
                window_supervised_ce=ce.tolist(), window_supervised_predicted_class=pred.tolist(), adjacent_js_values=js.tolist())
        result.append(row)
    return result


def anchor_row(row, expected):
    for key in ('sample_id', 'model', 'seed', 'target', 'speaker', 'input_sha256', 'prediction',
                'original_support_frames', 'stored_frames'):
        require(row[key] == expected[key], 'SSC clean anchor differs at '+key)
    require(np.isfinite(row['loss']) and row['loss'] >= 0 and row['correct'] == int(row['prediction'] == row['target']),
            'Invalid ordinary supervised result')
    return row['loss']-expected['loss']


def validate_windows(rows, spec, target=None):
    require([r['window_frames'] for r in rows] == spec['window_frames'], 'Window width coverage/order differs')
    for row in rows:
        for metric in METRICS:
            value = row[metric]
            require(value is None or np.isfinite(value), 'Nonfinite cached window metric')
        if row['valid_windows'] < 2:
            require(all(row[k] is None for k in METRICS) and row['undefined_reason'] == 'fewer_than_two_complete_windows',
                    'Insufficient windows must remain explicitly undefined')
            continue
        count = row['valid_windows']
        require(len(row['true_class_window_probability']) == len(row['window_predicted_class']) ==
                len(row['window_supervised_ce']) == len(row['window_supervised_predicted_class']) == count and
                len(row['adjacent_js_values']) == count-1, 'Cached window arrays have different coverage')
        require(0 <= row['adjacent_js_nats'] <= np.log(2)+1e-12 and
                all(0 <= row[k] <= 1 for k in ('true_class_mean_probability', 'window_accuracy',
                    'top_class_switch_fraction', 'window_supervised_accuracy')) and
                (row['true_class_concentration'] is None or 0 <= row['true_class_concentration'] <= 1) and
                row['window_mean_supervised_ce'] >= -1e-12, 'Window descriptor outside its range')
        require(np.isclose(np.mean(row['adjacent_js_values']), row['adjacent_js_nats'], atol=1e-12) and
                np.isclose(np.mean(row['window_supervised_ce']), row['window_mean_supervised_ce'], atol=1e-12),
                'Saved window arrays/means differ')
        mass = np.asarray(row['true_class_window_probability'])
        pred = np.asarray(row['window_predicted_class'])
        supervised_pred = np.asarray(row['window_supervised_predicted_class'])
        require(np.isfinite(mass).all() and np.all((mass >= 0) & (mass <= 1)) and
                np.isclose(mass.mean(), row['true_class_mean_probability'], atol=1e-12), 'Saved true-class evidence differs')
        if target is not None:
            require(np.isclose((pred == target).mean(), row['window_accuracy'], atol=1e-12) and
                    np.isclose((supervised_pred == target).mean(), row['window_supervised_accuracy'], atol=1e-12) and
                    np.isclose((pred[1:] != pred[:-1]).mean(), row['top_class_switch_fraction'], atol=1e-12),
                    'Saved window class arrays/accuracy/switches differ')
