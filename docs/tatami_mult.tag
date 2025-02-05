<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
<tagfile doxygen_version="1.12.0">
  <compound kind="file">
    <name>tatami_mult.hpp</name>
    <path>/github/workspace/include/tatami_mult/</path>
    <filename>tatami__mult_8hpp.html</filename>
    <class kind="struct">tatami_mult::Options</class>
    <namespace>tatami_mult</namespace>
  </compound>
  <compound kind="struct">
    <name>tatami_mult::Options</name>
    <filename>structtatami__mult_1_1Options.html</filename>
    <member kind="variable">
      <type>int</type>
      <name>num_threads</name>
      <anchorfile>structtatami__mult_1_1Options.html</anchorfile>
      <anchor>a5ea4a8dc6044006b4f64846429b9bbe8</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>prefer_larger</name>
      <anchorfile>structtatami__mult_1_1Options.html</anchorfile>
      <anchor>a03e0813cf53e087dc9112f91b670c1af</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>column_major_output</name>
      <anchorfile>structtatami__mult_1_1Options.html</anchorfile>
      <anchor>aaeda08a655cfd0b8ed4366aac50c1abb</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="namespace">
    <name>tatami_mult</name>
    <filename>namespacetatami__mult.html</filename>
    <class kind="struct">tatami_mult::Options</class>
    <member kind="function">
      <type>void</type>
      <name>multiply</name>
      <anchorfile>namespacetatami__mult.html</anchorfile>
      <anchor>a7eb0e957877357d3f3fdf106b7b1c0c3</anchor>
      <arglist>(const tatami::Matrix&lt; Value_, Index_ &gt; &amp;left, const Right_ *right, Output_ *output, const Options &amp;opt)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>multiply</name>
      <anchorfile>namespacetatami__mult.html</anchorfile>
      <anchor>a22b5ef50f3b28580216ed5b6d434dde0</anchor>
      <arglist>(const Left_ *left, const tatami::Matrix&lt; Value_, Index_ &gt; &amp;right, Output_ *output, const Options &amp;opt)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>multiply</name>
      <anchorfile>namespacetatami__mult.html</anchorfile>
      <anchor>a546ba58cf2b39a6d651c11570d839d5d</anchor>
      <arglist>(const tatami::Matrix&lt; Value_, Index_ &gt; &amp;left, const std::vector&lt; Right_ * &gt; &amp;right, const std::vector&lt; Output_ * &gt; &amp;output, const Options &amp;opt)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>multiply</name>
      <anchorfile>namespacetatami__mult.html</anchorfile>
      <anchor>a99dc6ea4d7c53f0d0ca78c9e884fe572</anchor>
      <arglist>(const std::vector&lt; Left_ * &gt; &amp;left, const tatami::Matrix&lt; Value_, Index_ &gt; &amp;right, const std::vector&lt; Output_ * &gt; &amp;output, const Options &amp;opt)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>multiply</name>
      <anchorfile>namespacetatami__mult.html</anchorfile>
      <anchor>ad5560680ed15301d2446e333c8cd9c56</anchor>
      <arglist>(const tatami::Matrix&lt; LeftValue_, LeftIndex_ &gt; &amp;left, const tatami::Matrix&lt; RightValue_, RightIndex_ &gt; &amp;right, Output_ *output, const Options &amp;opt)</arglist>
    </member>
  </compound>
  <compound kind="page">
    <name>index</name>
    <title>Matrix multiplication for tatami</title>
    <filename>index.html</filename>
    <docanchor file="index.html">md__2github_2workspace_2README</docanchor>
  </compound>
</tagfile>
