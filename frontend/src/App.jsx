import { useState, useEffect } from 'react';
import './index.css';

function App() {
  const [drugs, setDrugs] = useState([]);
  const [drugA, setDrugA] = useState('');
  const [drugB, setDrugB] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('phase3');

  useEffect(() => {
    fetch('http://localhost:8000/api/drugs')
      .then(res => res.json())
      .then(data => setDrugs(data.drugs || []))
      .catch(err => console.error("Failed to load drugs:", err));
  }, []);

  const handlePredict = async (e) => {
    e.preventDefault();
    if (!drugA || !drugB) {
      setError('Please select two different drugs.');
      return;
    }
    if (drugA === drugB) {
      setError('Please select two different drugs. You selected the same drug twice.');
      return;
    }
    
    setError(null);
    setLoading(true);
    setResults(null);
    
    try {
      const response = await fetch('http://localhost:8000/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ drug_a: drugA, drug_b: drugB })
      });
      
      if (!response.ok) throw new Error("Failed to fetch predictions from AI.");
      
      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-16 max-w-6xl">
      <header className="mb-16 text-center">
        <h1 className="text-6xl font-black mb-6 tracking-tight text-transparent bg-clip-text bg-gradient-to-br from-indigo-600 to-teal-500">
          FAERS Intelligence
        </h1>
        <p className="text-slate-500 text-xl max-w-2xl mx-auto font-medium leading-relaxed">
          Predict adverse events and estimate risk severity for drug-drug interactions using Machine Learning Models trained on millions of historical FDA reports.
        </p>
      </header>

      <div className="grid md:grid-cols-3 gap-10">
        
        {/* Left Sidebar: Search Panel */}
        <div className="md:col-span-1">
          <div className="glass-card border-t-4 border-t-indigo-500 relative overflow-hidden">
            <h2 className="text-2xl font-bold mb-8 text-slate-800 flex items-center tracking-tight">
              <span className="text-indigo-500 mr-3">🔍</span> Analyze Drugs
            </h2>
            
            <form onSubmit={handlePredict} className="space-y-6 relative z-10">
              <div>
                <label className="block text-sm font-bold text-slate-500 mb-2 uppercase tracking-wide">Drug A</label>
                <select 
                  className="glass-input"
                  value={drugA} 
                  onChange={(e) => setDrugA(e.target.value)}
                >
                  <option value="">Select a drug...</option>
                  {drugs.map(d => <option key={d} value={d}>{d}</option>)}
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-bold text-slate-500 mb-2 uppercase tracking-wide">Drug B</label>
                <select 
                  className="glass-input"
                  value={drugB} 
                  onChange={(e) => setDrugB(e.target.value)}
                >
                  <option value="">Select a drug...</option>
                  {drugs.map(d => <option key={d} value={d}>{d}</option>)}
                </select>
              </div>

              {error && <div className="text-rose-600 text-sm font-medium bg-rose-50 p-4 rounded-xl border border-rose-100 shadow-sm">{error}</div>}
              
              <button 
                type="submit" 
                disabled={loading || !drugA || !drugB}
                className={`w-full btn-primary flex justify-center items-center mt-4 ${loading ? 'opacity-75 cursor-not-allowed' : ''}`}
              >
                {loading ? (
                  <span className="flex items-center"><svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg> Analyzing...</span>
                ) : (
                  <span className="text-lg">Run AI Analysis</span>
                )}
              </button>
            </form>
          </div>
        </div>

        {/* Right Main Panel: Results Dashboard */}
        <div className="md:col-span-2">
          {!results && !loading && (
            <div className="h-full flex flex-col items-center justify-center text-slate-400 border-2 border-dashed border-slate-200 rounded-2xl p-16 text-center bg-white/50 backdrop-blur-sm shadow-sm hover:border-indigo-300 transition-colors">
              <span className="text-6xl mb-6 drop-shadow-sm">🔬</span>
              <h3 className="text-2xl font-bold mb-3 text-slate-600">Awaiting Input</h3>
              <p className="text-lg max-w-sm">Select two drugs and run the analysis to view historical evidence and AI predictions.</p>
            </div>
          )}

          {loading && (
            <div className="h-full flex items-center justify-center bg-white/50 backdrop-blur-sm rounded-2xl border border-slate-200 shadow-sm">
              <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-indigo-600"></div>
            </div>
          )}

          {results && (
            <div className="animate-fade-in-up">
              <h2 className="text-3xl font-black mb-8 flex items-center tracking-tight text-slate-800">
                Analysis: 
                <span className="text-teal-600 ml-3 bg-teal-50 px-3 py-1 rounded-lg border border-teal-200">{results.drug_a}</span> 
                <span className="text-slate-300 mx-3">+</span> 
                <span className="text-teal-600 bg-teal-50 px-3 py-1 rounded-lg border border-teal-200">{results.drug_b}</span>
              </h2>

              {/* Tabs */}
              <div className="flex space-x-2 mb-8 bg-slate-200/50 p-1.5 rounded-xl w-fit shadow-sm">
                <button
                  className={`px-5 py-2.5 font-bold text-sm rounded-lg transition-all shadow-sm flex items-center ${activeTab === 'phase3' ? 'bg-white text-indigo-600 border border-slate-200' : 'text-slate-500 hover:text-slate-700 hover:bg-white/50 border border-transparent'}`}
                  onClick={() => setActiveTab('phase3')}
                >
                  🕸️ Phase 3: PyTorch GNN
                </button>
                <button
                  className={`px-5 py-2.5 font-bold text-sm rounded-lg transition-all shadow-sm flex items-center ${activeTab === 'phase2' ? 'bg-white text-violet-600 border border-slate-200' : 'text-slate-500 hover:text-slate-700 hover:bg-white/50 border border-transparent'}`}
                  onClick={() => setActiveTab('phase2')}
                >
                  🤖 Phase 2: XGBoost
                </button>
                <button
                  className={`px-5 py-2.5 font-bold text-sm rounded-lg transition-all shadow-sm flex items-center ${activeTab === 'phase1' ? 'bg-white text-sky-600 border border-slate-200' : 'text-slate-500 hover:text-slate-700 hover:bg-white/50 border border-transparent'}`}
                  onClick={() => setActiveTab('phase1')}
                >
                  📊 Phase 1: FAERS Data
                </button>
              </div>

              {/* Tab Content */}
              {activeTab === 'phase3' && (
                <div className="glass-card border-t-4 border-t-indigo-500 relative overflow-hidden">
                  <div className="absolute top-0 right-0 w-32 h-32 bg-indigo-500/10 rounded-bl-full -mr-16 -mt-16 z-0"></div>
                  
                  <div className="relative z-10">
                    <h3 className="text-2xl font-black text-slate-800 mb-3 tracking-tight">Graph Neural Network Prediction</h3>
                    <p className="text-slate-500 text-base mb-8 max-w-lg leading-relaxed">
                      The state-of-the-art PyTorch Geometric model predicts interaction risk by analyzing the entire FDA drug network, allowing it to accurately assess rare or unrecorded interactions.
                    </p>
                    
                    <div className="bg-gradient-to-br from-slate-50 to-indigo-50/50 rounded-2xl p-8 border border-slate-200 flex flex-col items-center justify-center text-center shadow-sm">
                      <div className="text-indigo-500 uppercase tracking-widest text-xs font-black mb-4">Predicted GNN Severity Score</div>
                      {results.phase3_gnn.label === "NOT FOUND IN GRAPH" ? (
                        <div className="text-xl font-bold text-slate-400 mb-6 py-6">One of these drugs is not in the Graph Network</div>
                      ) : (
                        <>
                          <div className="text-7xl font-black text-slate-800 mb-6 drop-shadow-sm">
                            {results.phase3_gnn.score.toFixed(2)}
                          </div>
                          <div>
                            <span className={`risk-badge-${results.phase3_gnn.label.split(' ')[0].toLowerCase()} px-5 py-2 text-base`}>
                              {results.phase3_gnn.label}
                            </span>
                          </div>
                        </>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'phase2' && (
                <div className="glass-card border-t-4 border-t-violet-500 relative overflow-hidden">
                  <div className="absolute top-0 right-0 w-32 h-32 bg-violet-500/10 rounded-bl-full -mr-16 -mt-16 z-0"></div>
                  
                  <div className="relative z-10">
                    <h3 className="text-2xl font-black text-slate-800 mb-3 tracking-tight">XGBoost ML Prediction</h3>
                    <p className="text-slate-500 text-base mb-8 max-w-lg leading-relaxed">
                      The tabular Machine Learning model predicts risk by comparing target-encoded frequencies from the 1-Billion row baseline. It is very fast but lacks deep structural context.
                    </p>
                    
                    <div className="bg-gradient-to-br from-slate-50 to-violet-50/50 rounded-2xl p-8 border border-slate-200 flex flex-col items-center justify-center text-center shadow-sm">
                      <div className="text-violet-500 uppercase tracking-widest text-xs font-black mb-4">Predicted XGBoost Score</div>
                      <div className="text-7xl font-black text-slate-800 mb-6 drop-shadow-sm">
                        {results.phase2_xgb.score.toFixed(2)}
                      </div>
                      <div>
                        <span className={`risk-badge-${results.phase2_xgb.label.split(' ')[0].toLowerCase()} px-5 py-2 text-base`}>
                          {results.phase2_xgb.label}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'phase1' && (
                <div className="glass-card border-t-4 border-t-sky-400">
                  <h3 className="text-2xl font-black text-slate-800 mb-3 tracking-tight">Historical FAERS Evidence</h3>
                  <p className="text-slate-500 text-base mb-8 max-w-lg leading-relaxed">
                    Top 10 most significant adverse events reported for this combination in the FDA database, ranked by PRR risk score.
                  </p>
                  
                  {results.historical_evidence.length > 0 ? (
                    <div className="overflow-hidden rounded-xl border border-slate-200 shadow-sm">
                      <table className="w-full text-left border-collapse bg-white">
                        <thead>
                          <tr className="bg-slate-50 text-slate-500 text-xs uppercase tracking-widest border-b border-slate-200">
                            <th className="p-4 font-bold">Adverse Event</th>
                            <th className="p-4 font-bold text-right">Co-occurrences</th>
                            <th className="p-4 font-bold text-right">PRR Score</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100">
                          {results.historical_evidence.map((row, idx) => (
                            <tr key={idx} className="hover:bg-slate-50/80 transition-colors">
                              <td className="p-4 font-semibold text-slate-700 capitalize">{row.event.toLowerCase()}</td>
                              <td className="p-4 text-right">
                                <span className="bg-teal-50 text-teal-700 px-3 py-1 rounded-md font-mono font-bold text-sm border border-teal-100">{row.co_occurrences}</span>
                              </td>
                              <td className="p-4 text-right">
                                <span className="bg-amber-50 text-amber-700 px-3 py-1 rounded-md font-mono font-bold text-sm border border-amber-100">{row.PRR.toFixed(2)}</span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <div className="text-center p-12 bg-slate-50 rounded-2xl border border-slate-200 text-slate-500 font-medium">
                      <span className="text-4xl mb-4 block drop-shadow-sm">📭</span>
                      No significant historical evidence found for this specific combination in the database.
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
        
      </div>
      
      <style dangerouslySetInnerHTML={{__html: `
        @keyframes fadeInUp {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fade-in-up {
          animation: fadeInUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }
      `}} />
    </div>
  );
}

export default App;
