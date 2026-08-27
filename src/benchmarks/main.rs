use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;

const SEED: u64 = 123;
const FOLDS: usize = 5;

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
struct Pair {
    a: u32,
    b: u32,
}

fn canon(x: u32, y: u32) -> Pair {
    if x <= y {
        Pair { a: x, b: y }
    } else {
        Pair { a: y, b: x }
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }
    fn usize(&mut self, n: usize) -> usize {
        (self.next() as usize) % n.max(1)
    }
}

fn parse_pairs(path: &Path) -> Result<(Vec<String>, Vec<Pair>), String> {
    let f = fs::File::open(path).map_err(|e| format!("{path:?}: {e}"))?;
    let mut lines = BufReader::new(f).lines();
    let header = lines.next().ok_or("empty file")?.map_err(|e| e.to_string())?;
    let header = header.trim_start_matches('\u{feff}');
    let cols: Vec<String> = header.split([',', '\t']).map(|s| s.trim().to_string()).collect();
    let (i0, i1) = gene_cols(&cols).ok_or_else(|| format!("no gene columns in {header}"))?;
    let mut names = Vec::new();
    let mut id = HashMap::<String, u32>::new();
    let mut pairs = Vec::new();
    let intern = |s: &str, names: &mut Vec<String>, id: &mut HashMap<String, u32>| -> u32 {
        let s = s.trim().to_uppercase();
        if let Some(&g) = id.get(&s) {
            return g;
        }
        let n = names.len() as u32;
        id.insert(s.clone(), n);
        names.push(s);
        n
    };
    for line in lines {
        let line = line.map_err(|e| e.to_string())?;
        if line.is_empty() {
            continue;
        }
        let c: Vec<&str> = line.split([',', '\t']).collect();
        if c.len() <= i0.max(i1) {
            continue;
        }
        let ga = intern(c[i0], &mut names, &mut id);
        let gb = intern(c[i1], &mut names, &mut id);
        if ga != gb {
            pairs.push(canon(ga, gb));
        }
    }
    pairs.sort_by_key(|p| (p.a, p.b));
    pairs.dedup();
    Ok((names, pairs))
}

fn norm_col(s: &str) -> String {
    s.to_lowercase().replace([' ', '_', '-'], "")
}

fn gene_cols(cols: &[String]) -> Option<(usize, usize)> {
    let lower: Vec<String> = cols.iter().map(|s| norm_col(s)).collect();
    let keys = [
        ("gene1", "gene2"),
        ("genea", "geneb"),
        ("genea.name", "geneb.name"),
        ("xname", "yname"),
        ("officialsymbolgenea", "officialsymbolgeneb"),
        ("ncbigenea", "ncbigeneb"),
        ("symbol1", "symbol2"),
    ];
    for (a, b) in keys {
        let ia = lower.iter().position(|c| c == a);
        let ib = lower.iter().position(|c| c == b);
        if let (Some(i), Some(j)) = (ia, ib) {
            return Some((i, j));
        }
    }
    if cols.len() >= 2 {
        Some((0, 1))
    } else {
        None
    }
}

fn sample_negatives(n_genes: u32, pos: &[Pair], rng: &mut Rng) -> Vec<Pair> {
    let pos_set: HashSet<Pair> = pos.iter().copied().collect();
    let mut neg = Vec::with_capacity(pos.len());
    let mut guard = 0;
    while neg.len() < pos.len() && guard < pos.len() * 50 {
        guard += 1;
        let a = rng.usize(n_genes as usize) as u32;
        let b = rng.usize(n_genes as usize) as u32;
        if a == b {
            continue;
        }
        let p = canon(a, b);
        if pos_set.contains(&p) {
            continue;
        }
        neg.push(p);
    }
    neg.sort_by_key(|p| (p.a, p.b));
    neg.dedup();
    neg
}

fn gene_folds(n_genes: usize, fold: usize) -> HashSet<u32> {
    (0..n_genes)
        .filter(|g| g % FOLDS == fold)
        .map(|g| g as u32)
        .collect()
}

#[derive(Clone, Copy)]
enum Split {
    Cv1,
    Cv2,
    Cv3,
}

fn assign(split: Split, fold: usize, n_genes: usize, pairs: &[Pair]) -> (Vec<Pair>, Vec<Pair>) {
    match split {
        Split::Cv1 => {
            let mut train = Vec::new();
            let mut test = Vec::new();
            for (i, p) in pairs.iter().enumerate() {
                if i % FOLDS == fold {
                    test.push(*p);
                } else {
                    train.push(*p);
                }
            }
            (train, test)
        }
        Split::Cv2 | Split::Cv3 => {
            let held = gene_folds(n_genes, fold);
            let mut train = Vec::new();
            let mut test = Vec::new();
            for p in pairs {
                let ha = held.contains(&p.a);
                let hb = held.contains(&p.b);
                match split {
                    Split::Cv3 if !ha && !hb => train.push(*p),
                    Split::Cv3 if ha && hb => test.push(*p),
                    Split::Cv2 if !ha && !hb => train.push(*p),
                    Split::Cv2 if ha != hb => test.push(*p),
                    _ => {}
                }
            }
            (train, test)
        }
    }
}

fn degrees(train_pos: &[Pair]) -> HashMap<u32, i32> {
    let mut deg: HashMap<u32, i32> = HashMap::new();
    for p in train_pos {
        *deg.entry(p.a).or_insert(0) += 1;
        *deg.entry(p.b).or_insert(0) += 1;
    }
    deg
}

fn score_of(deg: &HashMap<u32, i32>, a: u32, b: u32) -> f64 {
    (*deg.get(&a).unwrap_or(&0) + *deg.get(&b).unwrap_or(&0)) as f64
}

fn auroc(y: &[u8], s: &[f64]) -> f64 {
    let mut idx: Vec<usize> = (0..y.len()).collect();
    idx.sort_by(|&i, &j| s[j].partial_cmp(&s[i]).unwrap_or(std::cmp::Ordering::Equal));
    let p = y.iter().filter(|&&v| v == 1).count() as f64;
    let n = y.len() as f64 - p;
    if p == 0.0 || n == 0.0 {
        return 0.5;
    }
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut auc = 0.0;
    let mut prev = f64::INFINITY;
    let mut prev_tp = 0.0;
    let mut prev_fp = 0.0;
    for i in idx {
        if s[i] != prev {
            auc += (fp - prev_fp) * (tp + prev_tp) / 2.0;
            prev = s[i];
            prev_tp = tp;
            prev_fp = fp;
        }
        if y[i] == 1 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
    }
    auc += (fp - prev_fp) * (tp + prev_tp) / 2.0;
    auc / (p * n)
}

fn aupr(y: &[u8], s: &[f64]) -> f64 {
    let p = y.iter().filter(|&&v| v == 1).count() as f64;
    if p == 0.0 {
        return 0.0;
    }
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in s {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    if hi == lo {
        return p / y.len() as f64;
    }
    let mut idx: Vec<usize> = (0..y.len()).collect();
    idx.sort_by(|&i, &j| s[j].partial_cmp(&s[i]).unwrap_or(std::cmp::Ordering::Equal));
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut area = 0.0;
    let mut prev_r = 0.0;
    for i in idx {
        if y[i] == 1 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        let rec = tp / p;
        let prec = tp / (tp + fp);
        area += (rec - prev_r) * prec;
        prev_r = rec;
    }
    area
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

fn run_split(name: &str, split: Split, n_genes: usize, pos: &[Pair], neg: &[Pair]) {
    let mut aucs = Vec::new();
    let mut aprs = Vec::new();
    let mut deg_zero = Vec::new();
    for fold in 0..FOLDS {
        let (tr_p, te_p) = assign(split, fold, n_genes, pos);
        let (tr_n, te_n) = assign(split, fold, n_genes, neg);
        if te_p.is_empty() || te_n.is_empty() {
            continue;
        }
        let pos_te: HashSet<Pair> = te_p.iter().copied().collect();
        let deg = degrees(&tr_p);
        let mut y = Vec::new();
        let mut s = Vec::new();
        let mut zeros = 0;
        let mut nte = 0;
        for p in te_p.iter().chain(te_n.iter()) {
            let lab = if pos_te.contains(p) { 1 } else { 0 };
            let sc = score_of(&deg, p.a, p.b);
            if sc == 0.0 {
                zeros += 1;
            }
            nte += 1;
            y.push(lab);
            s.push(sc);
        }
        aucs.push(auroc(&y, &s));
        aprs.push(aupr(&y, &s));
        deg_zero.push(zeros as f64 / nte as f64);
        let _ = tr_n;
    }
    println!(
        "{name}\tdegree\tAUROC={:.4}\tAUPR={:.4}\tzero_degree_frac={:.4}\tfolds={}",
        mean(&aucs),
        mean(&aprs),
        mean(&deg_zero),
        aucs.len()
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: slp-bench <pairs.csv>");
        std::process::exit(2);
    }
    let path = Path::new(&args[1]);
    let (names, pos) = parse_pairs(path).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    let n_genes = names.len() as u32;
    let mut rng = Rng(SEED);
    let neg = sample_negatives(n_genes, &pos, &mut rng);
    println!(
        "file={}\tgenes={}\tpos={}\tneg={}",
        path.display(),
        names.len(),
        pos.len(),
        neg.len()
    );
    run_split("pair-holdout", Split::Cv1, names.len(), &pos, &neg);
    run_split("one-new-gene", Split::Cv2, names.len(), &pos, &neg);
    run_split("two-new-gene", Split::Cv3, names.len(), &pos, &neg);
}
