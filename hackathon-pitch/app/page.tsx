export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 relative overflow-hidden">
      {/* Decorative Background Elements */}
      <div className="fixed inset-0 pointer-events-none font-mono">
        {/* Floating gradient orbs */}
        <div className="absolute top-20 left-10 w-72 h-72 bg-purple-500/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-20 right-10 w-96 h-96 bg-pink-500/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
        <div className="absolute top-1/2 left-1/2 w-64 h-64 bg-purple-400/5 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }}></div>

        {/* Code/Prompt Engineering symbols - static, no animation */}
        <div className="absolute top-1/4 left-8 text-8xl text-purple-400/10">{"{"}</div>
        <div className="absolute top-1/4 right-8 text-8xl text-pink-400/10">{"}"}</div>
        <div className="absolute bottom-1/3 left-16 text-6xl text-purple-300/8">{"<>"}</div>
        <div className="absolute bottom-1/4 right-24 text-6xl text-pink-300/8">{"[ ]"}</div>

        {/* Prompt engineering related symbols */}
        <div className="absolute top-1/3 left-1/4 text-7xl text-purple-400/8">#</div>
        <div className="absolute top-2/3 right-1/3 text-6xl text-pink-300/8">@</div>
        <div className="absolute top-1/2 left-20 text-5xl text-purple-300/10">$</div>
        <div className="absolute bottom-1/4 left-1/3 text-6xl text-pink-400/8">-&gt;</div>
        <div className="absolute top-40 right-1/4 text-5xl text-purple-400/8">...</div>
        <div className="absolute bottom-1/2 right-16 text-7xl text-pink-300/8">|</div>
        <div className="absolute top-1/2 right-1/3 text-5xl text-purple-300/10">*</div>
        <div className="absolute bottom-40 left-1/4 text-6xl text-pink-400/8">:</div>
        <div className="absolute top-1/3 right-20 text-4xl text-purple-300/12">&quot;prompt&quot;</div>
        <div className="absolute bottom-1/3 right-1/4 text-5xl text-pink-300/8">`</div>
        <div className="absolute top-1/4 left-1/3 text-4xl text-purple-400/10">def</div>
        <div className="absolute bottom-1/5 right-1/5 text-4xl text-pink-300/10">class</div>
        <div className="absolute top-3/4 left-1/5 text-5xl text-purple-300/8">=&gt;</div>
        <div className="absolute top-1/5 right-2/5 text-5xl text-pink-400/8">!=</div>
        <div className="absolute bottom-2/3 left-2/5 text-4xl text-purple-300/10">if</div>
        <div className="absolute top-2/5 right-1/5 text-6xl text-pink-300/8">//</div>
        <div className="absolute bottom-1/6 left-1/6 text-5xl text-purple-400/8">&&</div>
      </div>

      {/* Hero Section */}
      <section className="min-h-screen flex items-center justify-center px-8 relative">
        <div className="max-w-5xl mx-auto text-center relative">
          <h1 className="text-6xl md:text-8xl font-bold mb-8 leading-tight">
            <span className="block text-gray-200 text-5xl md:text-7xl mb-2 animate-[fade-in-up_0.8s_ease-out]">
              From Prompt Voodoo to
            </span>
            <span
              className="block text-transparent bg-clip-text bg-gradient-to-r from-purple-300 via-pink-400 to-purple-300 bg-[length:200%_auto] animate-[gradient-shift_3s_ease-in-out_infinite,glow-pulse_2s_ease-in-out_infinite,fade-in-up_0.8s_ease-out_0.3s_both]"
              style={{ textShadow: '0 0 40px rgba(168, 85, 247, 0.5)' }}
            >
              Production-Ready AI
            </span>
          </h1>
          <p className="text-xl md:text-2xl text-gray-100 mb-12 leading-relaxed animate-[fade-in-up_0.8s_ease-out_0.6s_both] max-w-4xl mx-auto">
            <span className="block">We help <span className="text-pink-300 font-semibold">AI Engineers and AI Project Leaders</span> build <span className="text-white font-semibold">higher quality AI solutions</span></span>
            <span className="block">by <span className="text-white font-semibold">automating</span> <span className="text-purple-200">tedious Prompt-Engineering</span>.</span>
          </p>
          <div className="flex justify-center gap-4 animate-[fade-in-up_0.8s_ease-out_0.9s_both]">
            <a href="#problem" className="px-8 py-4 bg-purple-600 hover:bg-purple-700 text-white rounded-full text-lg font-semibold transition-all transform hover:scale-105 hover:shadow-[0_0_30px_rgba(168,85,247,0.6)]">
              Learn More
            </a>
          </div>
        </div>
      </section>

      {/* Problem Section */}
      <section id="problem" className="min-h-screen flex items-center justify-center px-8 py-20 relative">
        <div className="max-w-5xl mx-auto relative">
          <h2 className="text-5xl md:text-6xl font-bold text-white mb-12">The Problem</h2>
          <div className="space-y-8">
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 border border-white/20">
              <h3 className="text-2xl font-bold text-purple-300 mb-4" style={{ textShadow: '0 0 1px rgba(0,0,0,0.8), 0 0 2px rgba(0,0,0,0.6)' }}>📊 Most AI solutions fail to get into Production</h3>
              <p className="text-xl text-gray-100 leading-relaxed">
                While it&apos;s trivial to build a prototype, it&apos;s notoriously hard to bring it into production
              </p>
            </div>

            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 border border-white/20">
              <h3 className="text-2xl font-bold text-purple-300 mb-4" style={{ textShadow: '0 0 1px rgba(0,0,0,0.8), 0 0 2px rgba(0,0,0,0.6)' }}>🎯 Systematic Evaluation Gap</h3>
              <p className="text-xl text-gray-100 leading-relaxed">
                Besides not measuring success and evaluating systematically, another significant reason is that it&apos;s hard to prompt engineer correctly
              </p>
            </div>

            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 border border-white/20">
              <h3 className="text-2xl font-bold text-purple-300 mb-4" style={{ textShadow: '0 0 1px rgba(0,0,0,0.8), 0 0 2px rgba(0,0,0,0.6)' }}>🔮 Prompt Engineering is Voodoo</h3>
              <p className="text-xl text-gray-100 leading-relaxed mb-4">
                It&apos;s not an exact science. It&apos;s kinda like magic or voodoo.
              </p>
              <p className="text-lg text-gray-200 italic">
                &quot;If you fail, 1000 kittens will die&quot; → accuracy magically increases 📈
              </p>
              <p className="text-lg text-gray-200 mt-2">
                But it can also counterintuitively decrease even if you think your prompt should be better
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Customer Section */}
      <section id="customer" className="min-h-screen flex items-center justify-center px-8 py-20 relative">
        <div className="max-w-5xl mx-auto relative">
          <h2 className="text-5xl md:text-6xl font-bold text-white mb-12">Our Customers</h2>
          <div className="grid md:grid-cols-2 gap-8">
            <div className="bg-gradient-to-br from-purple-600/20 to-pink-600/20 backdrop-blur-lg rounded-2xl p-8 border border-purple-500/30">
              <h3 className="text-2xl font-bold text-white mb-4">🏢 Who They Are</h3>
              <ul className="space-y-3 text-lg text-gray-100">
                <li>✓ Companies small and big wanting AI in production</li>
                <li>✓ Teams struggling with quality and accuracy</li>
                <li>✓ Developers tired of prompt engineering</li>
                <li>✓ Project leadership seeking measurable results</li>
              </ul>
            </div>

            <div className="bg-gradient-to-br from-pink-600/20 to-purple-600/20 backdrop-blur-lg rounded-2xl p-8 border border-pink-500/30">
              <h3 className="text-2xl font-bold text-white mb-4">📈 Market Opportunity</h3>
              <ul className="space-y-3 text-lg text-gray-100">
                <li>✓ Very hyped topic with lots of attention</li>
                <li>✓ Large and growing market</li>
                <li>✓ Early focus: Chat with your data (RAG)</li>
                <li>✓ Deep expertise in the space</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* Demo Section */}
      <section id="demo" className="min-h-screen flex items-center justify-center px-8 py-20 relative">
        <div className="max-w-5xl mx-auto relative">
          <h2 className="text-5xl md:text-6xl font-bold text-white mb-12">The Demo</h2>

          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 border border-white/20 mb-8">
            <h3 className="text-3xl font-bold text-purple-300 mb-4">Entry Point</h3>
            <p className="text-xl text-gray-100">
              Company has built a knowledge agent
            </p>
          </div>

          <div className="space-y-6">
            <h3 className="text-3xl font-bold text-white mb-6">User Journey</h3>

            <div className="bg-gradient-to-r from-purple-600/20 to-transparent backdrop-blur-lg rounded-2xl p-6 border-l-4 border-purple-500">
              <h4 className="text-xl font-bold text-purple-300 mb-3">Step 1: Dataset Creation</h4>
              <p className="text-lg text-gray-100 mb-3">Company needs a dataset of:</p>
              <ul className="space-y-2 text-gray-100 ml-6">
                <li>• Realistic questions</li>
                <li>• Expected sources where the info for the answer lies</li>
                <li>• Expected answers that the company would give without AI</li>
              </ul>
            </div>

            <div className="bg-gradient-to-r from-pink-600/20 to-transparent backdrop-blur-lg rounded-2xl p-6 border-l-4 border-pink-500">
              <h4 className="text-xl font-bold text-pink-300 mb-3">Step 2: Automated Training</h4>
              <p className="text-lg text-gray-100 mb-3">We use the dataset to &quot;train&quot; (automatically prompt engineer):</p>
              <ul className="space-y-2 text-gray-100 ml-6">
                <li>• Prompts are generated by AI</li>
                <li>• Then automatically tested against the dataset</li>
                <li>• Rinse, Repeat</li>
                <li className="font-bold text-purple-300">→ This improves answer accuracy in a measurable way without any manual prompt engineering</li>
              </ul>
            </div>

            <div className="bg-gradient-to-r from-purple-600/20 to-transparent backdrop-blur-lg rounded-2xl p-6 border-l-4 border-purple-500">
              <h4 className="text-xl font-bold text-purple-300 mb-3">Step 3: What We Show</h4>
              <ul className="space-y-2 text-gray-100 ml-6">
                <li>✓ Dataset</li>
                <li>✓ Train script</li>
                <li>✓ Prompt old vs new in the UI with one example question</li>
                <li>✓ Run eval script and compare results</li>
              </ul>
            </div>
          </div>

          <div className="mt-12 text-center">
            <div className="inline-block bg-gradient-to-r from-purple-600 to-pink-600 rounded-2xl p-8">
              <p className="text-3xl font-bold text-white">Any questions?</p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 text-center text-gray-300">
        <p>Automated Prompt Engineering for Production AI</p>
      </footer>
    </div>
  );
}
