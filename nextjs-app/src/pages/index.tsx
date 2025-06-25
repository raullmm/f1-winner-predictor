import { type NextPage } from "next";
import Head from "next/head";

import Docs from "src/components/docs";
import Predictor from "../components/predictor";

/**
 *  Home
 *  — Paleta corporativa azul (#5271ff)
 */
const Home: NextPage = () => {
  return (
    <>
      <Head>
        <title>F1 Race Predictor</title>
        <meta name="description" content="Predicting F1 races with ML" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className="flex flex-col md:flex-row min-h-screen font-inter">
        {/* Panel izquierdo — Predictor */}
        <section
          className="
            md:fixed md:left-0 md:h-full md:overflow-y-auto
            px-16 py-12 md:w-1/2 xl:w-1/3
            text-white
            bg-gradient-to-b from-[#5271ff] via-[#4363e6] to-[#2b48c8]
          "
        >
          <Predictor />
        </section>

        {/* Panel derecho — documentación / texto */}
        <section
          className="
            p-16 md:py-24 md:px-16 xl:px-28 md:w-1/2 xl:w-2/3
            md:fixed md:right-0 md:h-full md:overflow-y-auto
            bg-white dark:bg-[#0d1021]
          "
        >
          <article
            id="871047b8-2997-4a68-9c0f-53ade839e37d"
            className="page sans"
          >
            <Docs />
          </article>
        </section>
      </main>
    </>
  );
};

export default Home;

